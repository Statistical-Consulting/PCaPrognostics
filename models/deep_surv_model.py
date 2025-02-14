# from sklearn.base import BaseEstimator, RegressorMixin
# import numpy as np
# import pandas as pd
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# import torch
# from lifelines.utils import concordance_index
# from sklearn.utils.validation import check_X_y, check_is_fitted
# import logging
# from sklearn.model_selection import train_test_split

# logger = logging.getLogger(__name__)




import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import torch
from lifelines.utils import concordance_index
from sklearn.utils.validation import check_X_y, check_is_fitted
import logging
from sklearn.model_selection import train_test_split, LeaveOneGroupOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state
from sksurv.util import Surv
import os
import pickle
import matplotlib.pyplot as plt
from collections import defaultdict
import copy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
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
                nn.Dropout(dropout)
            ])
            prev_size = size

        # Output layer (1 node for hazard prediction)
        layers.append(nn.Linear(prev_size, 1, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class DeepSurvModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_features=None, hidden_layers=[16, 16], dropout=0.5,
                 learning_rate=0.01, device='cpu', random_state=123,
                 batch_size=128, num_epochs=100, patience=10):
        self.n_features = n_features
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.random_state = random_state
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted_ = False
        self.training_history_ = {'train_loss': [], 'val_loss': []}
        self.n_features_in_ = None

    def fit(self, X, y):
        # Input validation for X and y
        X, y = check_X_y(X, y, accept_sparse=True)

        # Set all seeds at start of fitting
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_state)
            torch.cuda.manual_seed_all(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.n_features_in_ = X.shape[1]
        self.init_network(self.n_features_in_)
        self.model.to(self.device)

        train_loader, val_loader = self._prepare_data(X, y, val_split=0.1)

        best_val_loss = float('inf')
        best_model_state = None
        counter = 0.0
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss_ = 0.0
            n_batches_ = 0
            for X_batch, time_batch, event_batch in train_loader:
                loss = self._train_step(X_batch, time_batch, event_batch)
                epoch_loss_ += loss
                n_batches_ += 1
            avg_train_loss = epoch_loss_ / n_batches_
            self.training_history_['train_loss'].append(avg_train_loss)

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, time_batch, event_batch in val_loader:
                    val_loss += self._eval_step(X_batch, time_batch, event_batch)

            val_loss = val_loss / len(val_loader)
            self.training_history_['val_loss'].append(val_loss)

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter > self.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, 'is_fitted_')
        if isinstance(X, pd.DataFrame):
            X = X.values
        X = torch.FloatTensor(X).to(self.device)
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
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "patience": self.patience
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def clone(self):
        super(self).clone()

    def _prepare_data(self, X, y, val_split = 0.1):
      # Ensure reproducible split
      X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=self.random_state)

      X_scaled_train = X_train
      times_train = np.ascontiguousarray(y_train['time']).astype(np.float32)
      event_field_train = 'status' if 'status' in y_train.dtype.names else 'event'
      events_train = np.ascontiguousarray(y_train[event_field_train]).astype(np.float32)
      X_tensor_train = torch.FloatTensor(X_scaled_train).to(self.device)
      time_tensor_train = torch.FloatTensor(times_train).to(self.device)
      event_tensor_train = torch.FloatTensor(events_train).to(self.device)

      X_scaled_val = X_val
      times_val = np.ascontiguousarray(y_val['time']).astype(np.float32)
      event_field_val = 'status' if 'status' in y_val.dtype.names else 'event'
      events_val = np.ascontiguousarray(y_val[event_field_val]).astype(np.float32)
      X_tensor_val = torch.FloatTensor(X_scaled_val).to(self.device)
      time_tensor_val = torch.FloatTensor(times_val).to(self.device)
      event_tensor_val = torch.FloatTensor(events_val).to(self.device)

      # Create DataLoader with reproducible generator
      train_dataset = TensorDataset(X_tensor_train, time_tensor_train, event_tensor_train)
      val_dataset = TensorDataset(X_tensor_val, time_tensor_val, event_tensor_val)

      generator = torch.Generator()
      generator.manual_seed(self.random_state)

      train_loader = DataLoader(
          train_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          generator=generator
      )

      val_loader = DataLoader(
          val_dataset,
          batch_size=self.batch_size,
          shuffle=True,
          generator=generator
      )

      return train_loader, val_loader

    def _negative_log_likelihood(self, risk_pred, times, events):
        _, idx = torch.sort(times, descending=True)
        risk_pred = risk_pred[idx]
        events = events[idx]
        log_risk = risk_pred
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
        self.optimizer.step()
        return loss.item()

    def _eval_step(self, X, times, events):
        risk_pred = self.model(X)
        loss = self._negative_log_likelihood(risk_pred, times, events)
        return loss.item()

    def _check_early_stopping(self, counter):
        if len(self.training_history_['val_loss']) < 2:
            return 0.0

        if self.training_history_['val_loss'][-1] < self.training_history_['val_loss'][-2]:
            counter = 0.0
        else:
            counter += 1.0
        return counter

    def c_index(self, risk_pred, y):
        if not isinstance(y, np.ndarray):
            y = y.detach().cpu().numpy()
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        time = y['time']
        event = y[event_field]
        if not isinstance(risk_pred, np.ndarray):
            risk_pred = risk_pred.detach().cpu().numpy()
        if np.isnan(risk_pred).all():
            return np.nan
        return concordance_index(time, risk_pred, event)

    def init_network(self, n_features):
        self.model = DeepSurvNet(n_features=n_features, hidden_layers=self.hidden_layers, dropout=self.dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)