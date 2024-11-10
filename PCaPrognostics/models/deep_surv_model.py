"""
Deep Survival Model Implementation using PyTorch

Features:
- Neural Network mit PyTorch
- Batch Training
- Early Stopping
- Negative Log Likelihood Loss
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import logging
from .base_model import BaseSurvivalModel

logger = logging.getLogger(__name__)

class SurvivalDataset(Dataset):
    """Dataset wrapper für Survival Daten"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        self.events = torch.FloatTensor(y[event_field])
        self.times = torch.FloatTensor(y['time'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.times[idx], self.events[idx]

class StandardizationLayer(nn.Module):
    """Custom Layer für Standardisierung"""
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(self, x):
        """Berechne mean und std auf Trainingsdaten"""
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True)
        # Vermeide Division durch 0
        self.std[self.std < 1e-8] = 1.0
        self.fitted = True

    def forward(self, x):
        if not self.fitted:
            return x
        return (x - self.mean) / self.std

class DeepSurvNet(nn.Module):
    """Neural Network für Survival Prediction mit integrierter Vorverarbeitung"""
    def __init__(self,
                 in_features,
                 hidden_layers=[128, 64, 32],
                 dropout=0.4):
        super().__init__()

        # Standardization layer
        self.standardize = StandardizationLayer()

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

        # Output layer (1 node für hazard prediction)
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # Standardize input
        x = self.standardize(x)
        return self.model(x)

class DeepSurvModel(BaseSurvivalModel):
    def __init__(self, config=None):
        super().__init__(config)
        self.use_pipeline = False  # Kein sklearn Pipeline
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

    def _negative_log_likelihood(self, risk_scores, times, events):
        """
        Berechnet negative log likelihood loss für Survival Prediction

        Parameters
        ----------
        risk_scores : torch.Tensor
            Predicted risk scores
        times : torch.Tensor
            Event times
        events : torch.Tensor
            Event indicators (1 wenn Event, 0 wenn zensiert)
        """
        # Sort by time
        _, idx = torch.sort(times, descending=True)
        risk_scores = risk_scores[idx]
        events = events[idx]

        # Cumulative sum of hazards
        hazard_ratio = torch.exp(risk_scores)
        log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))
        uncensored_likelihood = risk_scores.T - log_risk
        censored_likelihood = uncensored_likelihood * events

        # Return negative log likelihood
        return -torch.sum(censored_likelihood)

    def _fit_direct(self, X, y, validation_data=None,
                  hidden_layers=[128, 64, 32],
                  batch_size=64,
                  learning_rate=0.001,
                  n_epochs=100,
                  early_stopping=True,
                  patience=10,
                  dropout=0.4,
                  **kwargs):
        """Direktes Training ohne sklearn Pipeline"""
        # Create datasets
        train_dataset = SurvivalDataset(X, y)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        if validation_data is not None:
            val_dataset = SurvivalDataset(*validation_data)
            val_loader = DataLoader(
                val_dataset,
                batch_size=len(val_dataset)
            )

        # Initialize model
        self.model = DeepSurvNet(
            in_features=X.shape[1],
            hidden_layers=hidden_layers,
            dropout=dropout
        ).to(self.device)

        # Fit standardization on training data
        X_tensor = torch.FloatTensor(X.values).to(self.device)
        self.model.standardize.fit(X_tensor)

        # Optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

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
                n_val_batches = 0

                with torch.no_grad():
                    for val_X, val_times, val_events in val_loader:
                        val_X = val_X.to(self.device)
                        val_times = val_times.to(self.device)
                        val_events = val_events.to(self.device)

                        val_pred = self.model(val_X)
                        val_loss += self._negative_log_likelihood(
                            val_pred.squeeze(),
                            val_times,
                            val_events
                        ).item()
                        n_val_batches += 1

                avg_val_loss = val_loss / n_val_batches
                logger.info(f'Epoch {epoch+1}/{n_epochs}: '
                          f'Training Loss = {avg_train_loss:.4f}, '
                          f'Validation Loss = {avg_val_loss:.4f}')

                # Early stopping
                if early_stopping:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        # Save best model state
                        best_model_state = {
                            'model_state': self.model.state_dict(),
                            'epoch': epoch,
                            'val_loss': avg_val_loss
                        }
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            logger.info(f"Early stopping triggered at epoch {epoch+1}")
                            # Restore best model
                            self.model.load_state_dict(best_model_state['model_state'])
                            break
            else:
                logger.info(f'Epoch {epoch+1}/{n_epochs}: '
                          f'Training Loss = {avg_train_loss:.4f}')

        self.is_fitted = True
        return self.model

    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy().squeeze()

    def predict_survival_function(self, X):
        """Predict survival function

        Returns a function that can be evaluated at arbitrary times
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predicting")

        # Get risk scores
        risk_scores = self.predict(X)

        # Convert to survival probabilities
        def survival_function(times):
            return np.exp(-np.exp(risk_scores) * times)

        return survival_function

    def _save_direct_model(self, path, fname):
        """Save PyTorch model"""
        model_path = os.path.join(path, 'model', f"{fname}.pt")
        state_dict = {
            'model_state': self.model.state_dict(),
            'standardize_mean': self.model.standardize.mean,
            'standardize_std': self.model.standardize.std,
        }
        torch.save(state_dict, model_path)

    def load_model(self, path, fname):
        """Load PyTorch model"""
        model_path = os.path.join(path, 'model', f"{fname}.pt")
        state_dict = torch.load(model_path, map_location=self.device)

        # Recreate model architecture
        if not hasattr(self, 'model'):
            raise ValueError("Model architecture must be initialized before loading")

        self.model.load_state_dict(state_dict['model_state'])
        self.model.standardize.mean = state_dict['standardize_mean']
        self.model.standardize.std = state_dict['standardize_std']
        self.model.standardize.fitted = True
        self.is_fitted = True