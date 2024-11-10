import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from sklearn.preprocessing import StandardScaler
from .base_model import BaseSurvivalModel
from utils.resampling import DeepSurvNestedCV
from utils.evaluation import cindex_score

logger = logging.getLogger(__name__)


class DeepSurvNet(nn.Module):
    """Neural network architecture for DeepSurv"""

    def __init__(self, n_features, hidden_layers=[32, 16]):
        super().__init__()

        layers = []
        prev_size = n_features

        # Build hidden layers
        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(0.2)
            ])
            prev_size = size

        # Output layer (1 node for hazard prediction)
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepSurvModel(BaseSurvivalModel):
    """Deep Survival Neural Network Implementation"""

    def __init__(self, hidden_layers=[32, 16], learning_rate=0.001,
                 batch_size=64, num_epochs=100, device='cuda', random_state=42):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.random_state = random_state

        # Set seeds for reproducibility
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        # Initialize standard scaler
        self.scaler = StandardScaler()

        # Initialize network to None
        self.network = None

        # Initialize training history
        self.training_history_ = {'train_loss': [], 'val_loss': []}

    def _uses_pipeline(self):
        """Indicate that this is a direct training model"""
        return False

    def init_network(self, n_features):
        """Initialize the network architecture"""
        self.network = DeepSurvNet(
            n_features=n_features,
            hidden_layers=self.hidden_layers
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate
        )

    def _prepare_data(self, X, y):
        """Prepare data for PyTorch training"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Extract time and event data
        times = np.ascontiguousarray(y['time']).astype(np.float32)
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        events = np.ascontiguousarray(y[event_field]).astype(np.float32)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        time_tensor = torch.FloatTensor(times).to(self.device)
        event_tensor = torch.FloatTensor(events).to(self.device)

        return TensorDataset(X_tensor, time_tensor, event_tensor)

    def _negative_log_likelihood(self, risk_pred, times, events):
        """Custom loss function for survival prediction"""
        # Sort by descending time
        _, idx = torch.sort(times, descending=True)
        risk_pred = risk_pred[idx]
        events = events[idx]

        # Calculate loss
        log_risk = risk_pred
        risk = torch.exp(log_risk)
        cumsum_risk = torch.cumsum(risk, dim=0)
        log_cumsum_risk = torch.log(cumsum_risk + 1e-10)

        # Event-specific loss
        event_loss = events * (log_risk - log_cumsum_risk)
        return -torch.mean(event_loss)

    def _train_step(self, X, times, events):
        """Perform single training step"""
        self.optimizer.zero_grad()
        risk_pred = self.network(X)
        loss = self._negative_log_likelihood(risk_pred, times, events)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def fit(self, X, y, data_container=None, params_cv=None,
            use_cohort_cv=True, n_splits_inner=5, **kwargs):
        """Fit Deep Survival Network with optional CV"""
        logger.info("Starting DeepSurv training...")

        if params_cv:
            # Use nested cross-validation
            cv = DeepSurvNestedCV(
                n_splits_inner=n_splits_inner,
                use_cohort_cv=use_cohort_cv,
                random_state=self.random_state
            )

            groups = data_container.get_groups() if data_container else None

            self.cv_results_ = cv.fit(
                self,
                X=X,
                y=y,
                groups=groups,
                param_grid=params_cv
            )

            # Train final model with best parameters
            best_params = max(
                self.cv_results_['cv_results'],
                key=lambda x: x['test_score']
            )['best_params']

            self._fit_direct(X, y, **best_params)

        else:
            # Direct training without CV
            self._fit_direct(X, y, **kwargs)

        return self

    def _fit_direct(self, X, y, validation_data=None, **kwargs):
        """Direct training implementation"""
        # Initialize network if not already done
        if self.network is None:
            self.init_network(X.shape[1])

        # Update parameters if provided
        for param, value in kwargs.items():
            setattr(self, param, value)

        # Prepare data
        train_dataset = self._prepare_data(X, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Setup validation data if provided
        val_loader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = self._prepare_data(X_val, y_val)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

        # Training loop
        best_val_loss = float('inf')
        patience = 10
        no_improve = 0
        best_weights = None

        logger.info(f"Training on device: {self.device}")

        for epoch in range(self.num_epochs):
            # Training
            self.network.train()
            epoch_loss = 0
            n_batches = 0

            for X_batch, time_batch, event_batch in train_loader:
                loss = self._train_step(X_batch, time_batch, event_batch)
                epoch_loss += loss
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches
            self.training_history_['train_loss'].append(avg_train_loss)

            # Validation
            if val_loader is not None:
                self.network.eval()
                val_loss = 0
                n_val_batches = 0

                with torch.no_grad():
                    for X_batch, time_batch, event_batch in val_loader:
                        risk_pred = self.network(X_batch)
                        loss = self._negative_log_likelihood(
                            risk_pred, time_batch, event_batch
                        )
                        val_loss += loss.item()
                        n_val_batches += 1

                avg_val_loss = val_loss / n_val_batches
                self.training_history_['val_loss'].append(avg_val_loss)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_weights = self.network.state_dict().copy()
                    no_improve = 0
                else:
                    no_improve += 1

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
                        f"val_loss={avg_val_loss:.4f}"
                    )

                if no_improve >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")
                best_weights = self.network.state_dict().copy()

        # Load best weights
        if best_weights is not None:
            self.network.load_state_dict(best_weights)

        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict risk scores for samples in X"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.network.eval()
        with torch.no_grad():
            risk_scores = self.network(X_tensor).cpu().numpy()

        return risk_scores.flatten()

    def _save_model(self, path, fname):
        """Save model specific files"""
        model_path = os.path.join(path, f"{fname}_model.pt")
        scaler_path = os.path.join(path, f"{fname}_scaler.pkl")
        history_path = os.path.join(path, f"{fname}_history.pkl")

        # Save PyTorch model
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'hidden_layers': self.hidden_layers,
            'n_features': next(self.network.parameters()).shape[1]
        }, model_path)

        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        # Save training history
        with open(history_path, 'wb') as f:
            pickle.dump(self.training_history_, f)

        logger.info(f"Model saved to {path}")

    def get_params(self, deep=True):
        """Get parameters for this estimator"""
        return {
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'device': self.device,
            'random_state': self.random_state
        }

    def set_params(self, **params):
        """Set parameters for this estimator"""
        for key, value in params.items():
            setattr(self, key, value)
        return self