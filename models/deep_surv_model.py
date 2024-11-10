import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import logging
from .base_model import BaseSurvivalModel
from utils.resampling import NestedResamplingCV
from utils.evaluation import cindex_score

logger = logging.getLogger(__name__)


class SurvivalDataset(Dataset):
    """Dataset wrapper for survival data"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        # Convert structured array to contiguous arrays
        self.times = torch.FloatTensor(y['time'].copy())  # Make contiguous copy
        if 'status' in y.dtype.names:
            self.events = torch.FloatTensor(y['status'].copy())
        else:
            self.events = torch.FloatTensor(y['event'].copy())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.times[idx], self.events[idx]


class DeepSurvNet(nn.Module):
    """Neural network for survival prediction"""

    def __init__(self, in_features, hidden_layers=[64, 32], dropout=0.4):
        super().__init__()

        # Input standardization
        self.register_buffer('mean', None)
        self.register_buffer('std', None)

        layers = []
        prev_size = in_features

        # Hidden layers
        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                nn.BatchNorm1d(size),
                nn.Dropout(dropout)
            ])
            prev_size = size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)
        self.is_standardized = False

    def fit_standardization(self, x):
        """Compute mean and std on training data"""
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True)
        self.std[self.std < 1e-8] = 1.0
        self.is_standardized = True

    def forward(self, x):
        if self.is_standardized:
            x = (x - self.mean) / self.std
        return self.model(x)


class DeepSurvModel(BaseSurvivalModel):
    """Deep survival model implementation"""

    def __init__(self, use_nested_cv=False):
        super().__init__()
        self.use_pipeline = False
        self.use_nested_cv = use_nested_cv
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def get_params(self, deep=True):
        """Get parameters for CV"""
        return {
            'use_nested_cv': self.use_nested_cv
        }

    def set_params(self, **parameters):
        """Set parameters for CV"""
        if 'use_nested_cv' in parameters:
            self.use_nested_cv = parameters['use_nested_cv']
        return self

    def fit(self, X, y, data_container=None, param_grid=None, **fit_params):
        """Enhanced fit method with optional nested CV"""
        if self.use_nested_cv and param_grid is not None:
            return self._fit_nested_cv(X, y, data_container, param_grid)
        else:
            # Original direct fitting logic
            return super().fit(X, y, data_container=data_container, **fit_params)

    def _fit_nested_cv(self, X, y, data_container, param_grid):
        """Nested CV implementation for DeepSurv"""
        try:
            logger.info("Starting nested CV for DeepSurv...")

            # Get groups for cohort-based CV
            groups = data_container.get_groups() if data_container else None

            # Initialize nested CV
            cv = NestedResamplingCV(
                n_splits_inner=5,
                use_cohort_cv=True,
                random_state=42
            )

            # Run nested CV
            self.cv_results_ = cv.fit(
                estimator=self,
                X=X,
                y=y,
                groups=groups,
                param_grid=param_grid,
                scoring=cindex_score
            )

            # Fit final model with best params on full dataset
            best_params = self.cv_results_['fold_results'][0]['best_params']
            self._fit_direct(X, y, **best_params)

            logger.info("Nested CV completed successfully")
            return self

        except Exception as e:
            logger.error(f"Error in nested CV: {str(e)}")
            raise

    def _fit_direct(self, X, y, validation_data=None,
                    hidden_layers=[64, 32],
                    batch_size=64,
                    learning_rate=0.001,
                    n_epochs=100,
                    early_stopping=True,
                    patience=10,
                    dropout=0.4):
        """Train the deep survival model"""

        # Create datasets
        train_dataset = SurvivalDataset(X, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if validation_data is not None:
            X_val, y_val = validation_data
            val_dataset = SurvivalDataset(X_val, y_val)
            val_loader = DataLoader(val_dataset, batch_size=len(val_dataset))

        # Initialize model
        self.model = DeepSurvNet(
            in_features=X.shape[1],
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(self.device)

        # Compute standardization on training data
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        self.model.fit_standardization(X_tensor)

        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        logger.info(f"Starting training for {n_epochs} epochs...")

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            n_batches = 0

            for batch_X, batch_times, batch_events in train_loader:
                batch_X = batch_X.to(self.device)
                batch_times = batch_times.to(self.device)
                batch_events = batch_events.to(self.device)

                optimizer.zero_grad()
                risk_scores = self.model(batch_X)
                loss = self._negative_log_likelihood(
                    risk_scores.squeeze(),
                    batch_times,
                    batch_events
                )
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / n_batches

            # Validation
            if validation_data is not None:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for val_X, val_times, val_events in val_loader:
                        val_X = val_X.to(self.device)
                        val_times = val_times.to(self.device)
                        val_events = val_events.to(self.device)

                        val_pred = self.model(val_X)
                        val_loss = self._negative_log_likelihood(
                            val_pred.squeeze(),
                            val_times,
                            val_events
                        ).item()

                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = self.model.state_dict()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping at epoch {epoch + 1}")
                            self.model.load_state_dict(best_model_state)
                            break

                logger.info(f'Epoch {epoch + 1}/{n_epochs}: '
                            f'Train Loss = {avg_train_loss:.4f}, '
                            f'Val Loss = {val_loss:.4f}')
            else:
                logger.info(f'Epoch {epoch + 1}/{n_epochs}: '
                            f'Train Loss = {avg_train_loss:.4f}')

        return self.model

    def _negative_log_likelihood(self, risk_scores, times, events):
        """Compute negative log likelihood loss"""
        # Sort by time
        _, idx = torch.sort(times, descending=True)
        risk_scores = risk_scores[idx]
        events = events[idx]

        # Compute loss
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_scores.T - log_risk
        censored_likelihood = uncensored_likelihood * events
        return -torch.sum(censored_likelihood)

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().squeeze()

    def _save_model(self, path, fname):
        """Save PyTorch model"""
        model_path = os.path.join(path, f"{fname}.pt")
        torch.save({
            'model_state': self.model.state_dict(),
            'mean': self.model.mean,
            'std': self.model.std,
        }, model_path)
        logger.info(f"Saved model to {model_path}")