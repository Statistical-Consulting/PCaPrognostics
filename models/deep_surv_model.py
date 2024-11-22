from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch
from lifelines.utils import concordance_index
from sklearn.utils.validation import check_X_y, check_is_fitted
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class DeepSurvNet(nn.Module):
    """Neural network architecture for DeepSurv"""

    def __init__(self, n_features, hidden_layers=[32, 16], dropout=0.2):
        super().__init__()
        layers = []
        prev_size = n_features
        self.model = None

        # Build hidden layers
        for size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, size),
                nn.ReLU(),
                # BatchNorm1d nur bei größeren Batches verwenden
                nn.Dropout(dropout)
            ])
            prev_size = size

        # Output layer (1 node for hazard prediction)
        layers.append(nn.Linear(prev_size, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepSurvModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=None, hidden_layers=[16, 16], dropout=0.5,
                 learning_rate=0.01, device='cpu', random_state=123):
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted_ = False
        self.training_history_ = {'train_loss': [], 'val_loss': []}
        self.n_features_in_ = None

    def fit(self, X, y, num_epochs=10):
        # Input validation for X and y
        X, y = check_X_y(X, y, accept_sparse=True)
        
        self.n_features_in_ = X.shape[1]
        self.init_network(self.n_features_in_)
        self.model.to(self.device)
        
        # Prepare and scale data
        train_dataset_, val_dataset_ = self._prepare_data(X, y, val_split = 0.1)
        train_loader_ = DataLoader(train_dataset_, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset_, batch_size = 32, shuffle = True)
        
        # Training loop
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss_ = 0.0
            n_batches_ = 0
            for X_batch, time_batch, event_batch in train_loader_:
                loss = self._train_step(X_batch, time_batch, event_batch)
                epoch_loss_ += loss
                n_batches_ += 1
            avg_train_loss = epoch_loss_ / n_batches_
            self.training_history_['train_loss'].append(avg_train_loss)
            
            # enter validation mode
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, time_batch, event_batch in val_loader:
                    val_loss += self._eval_step(X_batch, time_batch, event_batch)

            val_loss = val_loss / len(val_loader)
            print(f"Epoch {epoch+1}, Train Loss; {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        X = torch.FloatTensor(self.scaler.transform(X)).to(self.device)
        self.model.eval()
        with torch.no_grad():
            risk_scores = self.model(X).cpu().numpy()
        return risk_scores.flatten()

    def score(self, X, y):
        check_is_fitted(self, 'is_fitted_')
        preds = self.predict(X)
        return self.c_index(-preds, y)

    def get_params(self, deep=True):
        return {
            "n_features": self.n_features,
            "hidden_layers": self.hidden_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "device": self.device,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
    def clone(self): 
        super(self).clone()

    def _prepare_data(self, X, y, val_split = 0.1):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)
        
        X_scaled_train = self.scaler.fit_transform(X_train)
        times_train = np.ascontiguousarray(y_train['time']).astype(np.float32)
        event_field_train = 'status' if 'status' in y_train.dtype.names else 'event'
        events_train = np.ascontiguousarray(y_train[event_field_train]).astype(np.float32)
        X_tensor_train = torch.FloatTensor(X_scaled_train).to(self.device)
        time_tensor_train = torch.FloatTensor(times_train).to(self.device)
        event_tensor_train = torch.FloatTensor(events_train).to(self.device)

        X_scaled_val = self.scaler.transform(X_val)
        times_val = np.ascontiguousarray(y_val['time']).astype(np.float32)
        event_field_val = 'status' if 'status' in y_val.dtype.names else 'event'
        events_val = np.ascontiguousarray(y_val[event_field_val]).astype(np.float32)
        X_tensor_val = torch.FloatTensor(X_scaled_val).to(self.device)
        time_tensor_val = torch.FloatTensor(times_val).to(self.device)
        event_tensor_val = torch.FloatTensor(events_val).to(self.device)

        
        return TensorDataset(X_tensor_train, time_tensor_train, event_tensor_train), TensorDataset(X_tensor_val, time_tensor_val, event_tensor_val)

    def _negative_log_likelihood(self, risk_pred, times, events):
        _, idx = torch.sort(times, descending=True)
        risk_pred = risk_pred[idx]
        events = events[idx]
        log_risk = risk_pred
        #print("Risk predictions before exp:", risk_pred)
        risk = torch.exp(log_risk)
        cumsum_risk = torch.cumsum(risk, dim=0)
        log_cumsum_risk = torch.log(cumsum_risk + 1e-10)
        event_loss = events * (log_risk - log_cumsum_risk)
        return -torch.mean(event_loss)

    def _train_step(self, X, times, events):
        self.optimizer.zero_grad()
        risk_pred = self.model(X)
        loss = self._negative_log_likelihood(risk_pred, times, events)
        loss.backward()
        #print([param.grad.norm().item() for param in self.model.parameters() if param.grad is not None])

        self.optimizer.step()
        return loss.item()
    
    def _eval_step(self, X, times, events): 
        risk_pred = self.model(X)
        loss = self._negative_log_likelihood(risk_pred, times, events)
        return loss.item()
        

    def c_index(self, risk_pred, y):
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy()
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        time = y['time']
        event = y[event_field]
        if not isinstance(risk_pred, np.ndarray):
            risk_pred = risk_pred.detach().cpu().numpy()
        return concordance_index(time, risk_pred, event)

    def init_network(self, n_features):
        self.model = DeepSurvNet(n_features=n_features, hidden_layers=self.hidden_layers).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)



################################################################################################################### Old implementation
# import os
# import pickle
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# import logging
# from sklearn.preprocessing import StandardScaler
# from .base_model import BaseSurvivalModel
# from utils.resampling import DeepSurvNestedCV
# from utils.evaluation import cindex_score

# logger = logging.getLogger(__name__)


# class DeepSurvNet(nn.Module):
#     """Neural network architecture for DeepSurv"""

#     def __init__(self, n_features, hidden_layers=[32, 16], dropout=0.2):
#         super().__init__()

#         layers = []
#         prev_size = n_features

#         # Build hidden layers
#         for size in hidden_layers:
#             layers.extend([
#                 nn.Linear(prev_size, size),
#                 nn.ReLU(),
#                 # BatchNorm1d nur bei größeren Batches verwenden
#                 nn.Dropout(dropout)
#             ])
#             prev_size = size

#         # Output layer (1 node for hazard prediction)
#         layers.append(nn.Linear(prev_size, 1))

#         self.model = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.model(x)


# class DeepSurvModel(BaseSurvivalModel):
#     """Deep Survival Neural Network Implementation"""

#     def __init__(self, hidden_layers=[32, 16], learning_rate=0.001,
#                  batch_size=64, num_epochs=100, device='cuda', random_state=42):
#         super().__init__()
#         self.hidden_layers = hidden_layers
#         self.learning_rate = learning_rate
#         self.batch_size = batch_size
#         self.num_epochs = num_epochs
#         self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
#         self.random_state = random_state

#         # Set seeds for reproducibility
#         torch.manual_seed(random_state)
#         np.random.seed(random_state)

#         # Initialize standard scaler
#         self.scaler = StandardScaler()

#         # Initialize network to None
#         self.network = None

#         # Initialize training history
#         self.training_history_ = {'train_loss': [], 'val_loss': []}

#     def _uses_pipeline(self):
#         """Indicate that this is a direct training model"""
#         return False

#     def init_network(self, n_features):
#         """Initialize the network architecture"""
#         self.network = DeepSurvNet(
#             n_features=n_features,
#             hidden_layers=self.hidden_layers
#         ).to(self.device)
#         self.optimizer = torch.optim.Adam(
#             self.network.parameters(),
#             lr=self.learning_rate
#         )

#     def _prepare_data(self, X, y):
#         """Prepare data for PyTorch training"""
#         # Scale features
#         X_scaled = self.scaler.fit_transform(X)

#         # Extract time and event data
#         times = np.ascontiguousarray(y['time']).astype(np.float32)
#         event_field = 'status' if 'status' in y.dtype.names else 'event'
#         events = np.ascontiguousarray(y[event_field]).astype(np.float32)

#         # Convert to tensors
#         X_tensor = torch.FloatTensor(X_scaled).to(self.device)
#         time_tensor = torch.FloatTensor(times).to(self.device)
#         event_tensor = torch.FloatTensor(events).to(self.device)

#         return TensorDataset(X_tensor, time_tensor, event_tensor)

#     def _negative_log_likelihood(self, risk_pred, times, events):
#         """Custom loss function for survival prediction"""
#         # Sort by descending time
#         _, idx = torch.sort(times, descending=True)
#         risk_pred = risk_pred[idx]
#         events = events[idx]

#         # Calculate loss
#         log_risk = risk_pred
#         risk = torch.exp(log_risk)
#         cumsum_risk = torch.cumsum(risk, dim=0)
#         log_cumsum_risk = torch.log(cumsum_risk + 1e-10)

#         # Event-specific loss
#         event_loss = events * (log_risk - log_cumsum_risk)
#         return -torch.mean(event_loss)

#     def _train_step(self, X, times, events):
#         """Perform single training step"""
#         self.optimizer.zero_grad()
#         risk_pred = self.network(X)
#         loss = self._negative_log_likelihood(risk_pred, times, events)
#         loss.backward()
#         self.optimizer.step()
#         return loss.item()

#     def fit(self, X, y, data_container=None, params_cv=None,
#             use_cohort_cv=True, n_splits_inner=5, **kwargs):
#         """Fit Deep Survival Network with optional CV"""
#         logger.info("Starting DeepSurv training...")

#         if params_cv:
#             # Use nested cross-validation
#             cv = DeepSurvNestedCV(
#                 n_splits_inner=n_splits_inner,
#                 use_cohort_cv=use_cohort_cv,
#                 random_state=self.random_state
#             )

#             groups = data_container.get_groups() if data_container else None

#             self.cv_results_ = cv.fit(
#                 self,
#                 X=X,
#                 y=y,
#                 groups=groups,
#                 param_grid=params_cv
#             )

#             # Train final model with best parameters
#             best_params = max(
#                 self.cv_results_['cv_results'],
#                 key=lambda x: x['test_score']
#             )['best_params']

#             self._fit_direct(X, y, **best_params)

#         else:
#             # Direct training without CV
#             self._fit_direct(X, y, **kwargs)

#         return self

#     def _fit_direct(self, X, y, validation_data=None, **kwargs):
#         """Direct training implementation"""
#         # Initialize network if not already done
#         if self.network is None:
#             self.init_network(X.shape[1])

#         # Update parameters if provided
#         for param, value in kwargs.items():
#             setattr(self, param, value)

#         # Prepare data
#         train_dataset = self._prepare_data(X, y)
#         train_loader = DataLoader(
#             train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True
#         )

#         # Setup validation data if provided
#         val_loader = None
#         if validation_data is not None:
#             X_val, y_val = validation_data
#             val_dataset = self._prepare_data(X_val, y_val)
#             val_loader = DataLoader(
#                 val_dataset,
#                 batch_size=self.batch_size,
#                 shuffle=False
#             )

#         # Training loop
#         best_val_loss = float('inf')
#         patience = 10
#         no_improve = 0
#         best_weights = None

#         logger.info(f"Training on device: {self.device}")

#         for epoch in range(self.num_epochs):
#             # Training
#             self.network.train()
#             epoch_loss = 0
#             n_batches = 0

#             for X_batch, time_batch, event_batch in train_loader:
#                 loss = self._train_step(X_batch, time_batch, event_batch)
#                 epoch_loss += loss
#                 n_batches += 1

#             avg_train_loss = epoch_loss / n_batches
#             self.training_history_['train_loss'].append(avg_train_loss)

#             # Validation
#             if val_loader is not None:
#                 self.network.eval()
#                 val_loss = 0
#                 n_val_batches = 0

#                 with torch.no_grad():
#                     for X_batch, time_batch, event_batch in val_loader:
#                         risk_pred = self.network(X_batch)
#                         loss = self._negative_log_likelihood(
#                             risk_pred, time_batch, event_batch
#                         )
#                         val_loss += loss.item()
#                         n_val_batches += 1

#                 avg_val_loss = val_loss / n_val_batches
#                 self.training_history_['val_loss'].append(avg_val_loss)

#                 if avg_val_loss < best_val_loss:
#                     best_val_loss = avg_val_loss
#                     best_weights = self.network.state_dict().copy()
#                     no_improve = 0
#                 else:
#                     no_improve += 1

#                 if epoch % 10 == 0:
#                     logger.info(
#                         f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, "
#                         f"val_loss={avg_val_loss:.4f}"
#                     )

#                 if no_improve >= patience:
#                     logger.info(f"Early stopping at epoch {epoch}")
#                     break
#             else:
#                 if epoch % 10 == 0:
#                     logger.info(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}")
#                 best_weights = self.network.state_dict().copy()

#         # Load best weights
#         if best_weights is not None:
#             self.network.load_state_dict(best_weights)

#         self.is_fitted = True
#         return self

#     def predict(self, X):
#         """Predict risk scores for samples in X"""
#         if not self.is_fitted or self.network is None:
#             raise ValueError("Model must be fitted before predicting")

#         # Handle scaling
#         X_scaled = self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else self.scaler.fit_transform(X)
#         X_tensor = torch.FloatTensor(X_scaled).to(self.device)

#         self.network.eval()
#         with torch.no_grad():
#             risk_scores = self.network(X_tensor).cpu().numpy()

#         return risk_scores.flatten()

#     def _save_model(self, path, fname):
#         """Save model specific files"""
#         model_path = os.path.join(path, f"{fname}_model.pt")
#         scaler_path = os.path.join(path, f"{fname}_scaler.pkl")
#         history_path = os.path.join(path, f"{fname}_history.pkl")

#         # Save PyTorch model
#         torch.save({
#             'model_state_dict': self.network.state_dict(),
#             'hidden_layers': self.hidden_layers,
#             'n_features': next(self.network.parameters()).shape[1]
#         }, model_path)

#         # Save scaler
#         with open(scaler_path, 'wb') as f:
#             pickle.dump(self.scaler, f)

#         # Save training history
#         with open(history_path, 'wb') as f:
#             pickle.dump(self.training_history_, f)

#         logger.info(f"Model saved to {path}")

#     def get_params(self, deep=True):
#         """Get parameters for this estimator"""
#         return {
#             'hidden_layers': self.hidden_layers,
#             'learning_rate': self.learning_rate,
#             'batch_size': self.batch_size,
#             'num_epochs': self.num_epochs,
#             'device': self.device,
#             'random_state': self.random_state
#         }

#     def set_params(self, **params):
#         """Set parameters for this estimator"""
#         for key, value in params.items():
#             setattr(self, key, value)
#         return self