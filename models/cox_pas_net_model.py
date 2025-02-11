from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import torch
from sklearn.utils.validation import check_X_y, check_is_fitted
import logging
from sklearn.model_selection import train_test_split
import math
import torch.optim as optim
import copy
from scipy.interpolate import interp1d
dtype = torch.FloatTensor


logger = logging.getLogger(__name__)

"""
The following code is based on the implementation from https://github.com/DataX-JieHao/Cox-PASNet
"""

def dropout_mask(n_node, drop_p):
	'''Construct a binary matrix to randomly drop nodes in a layer.
	Input:
		n_node: number of nodes in the layer.
		drop_p: the probability that a node is to be dropped.
	Output:
		mask: a binary matrix, where 1 --> keep the node; 0 --> drop the node.
	'''
	keep_p = 1.0 - drop_p
	mask = torch.Tensor(np.random.binomial(1, keep_p, size=n_node))
	###if gpu is being used
	if torch.cuda.is_available():
		mask = mask.cuda()
	###
	return mask

def s_mask(sparse_level, param_matrix, nonzero_param_1D, dtype):
	'''Construct a binary matrix w.r.t. a sparsity level of weights between two consecutive layers
	Input:
		sparse_level: a percentage value in [0, 100) represents the proportion of weights in a sub-network to be dropped.
		param_matrix: a weight matrix for entrie network.
		nonzero_param_1D: 1D of non-zero 'param_matrix' (which is the weights selected from a sub-network).
		dtype: define the data type of tensor (i.e. dtype=torch.FloatTensor).
	Output:
		param_mask: a binary matrix, where 1 --> keep the node; 0 --> drop the node.
	'''
	###take the absolute values of param_1D
	non_neg_param_1D = torch.abs(nonzero_param_1D)
	###obtain the number of params
	num_param = nonzero_param_1D.size(0)
	###obtain the kth number based on sparse_level
	top_k = math.ceil(num_param*(100-sparse_level)*0.01)
	###obtain the k largest params
	sorted_non_neg_param_1D, indices = torch.topk(non_neg_param_1D, top_k)
	param_mask = torch.abs(param_matrix) > sorted_non_neg_param_1D.min()
	param_mask = param_mask.type(dtype)
	###if gpu is being used
	if torch.cuda.is_available():
		param_mask = param_mask.cuda()
	###
	return param_mask

def R_set(x):
	'''Create an indicator matrix of risk sets, where T_j >= T_i.
	Note that the input data have been sorted in descending order.
	Input:
		x: a PyTorch tensor that the number of rows is equal to the number of samples.
	Output:
		indicator_matrix: an indicator matrix (which is a lower traiangular portions of matrix).
	'''
	n_sample = x.size(0)
	matrix_ones = torch.ones(n_sample, n_sample)
	indicator_matrix = torch.tril(matrix_ones)

	return(indicator_matrix)


def neg_par_log_likelihood(pred, ytime, yevent):
	'''Calculate the average Cox negative partial log-likelihood.
	Note that this function requires the input data have been sorted in descending order.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		cost: the cost that is to be minimized.
	'''
	n_observed = yevent.sum(0)
	ytime_indicator = R_set(ytime)
	###if gpu is being used
	if torch.cuda.is_available():
		ytime_indicator = ytime_indicator.cuda()
	###
	risk_set_sum = ytime_indicator.mm(torch.exp(pred))
	diff = pred - torch.log(risk_set_sum)
	sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
	cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))

	return(cost)


def c_index(pred, ytime, yevent):
	'''Calculate concordance index to evaluate models.
	Input:
		pred: linear predictors from trained model.
		ytime: true survival time from load_data().
		yevent: true censoring status from load_data().
	Output:
		concordance_index: c-index (between 0 and 1).
	'''
	n_sample = len(ytime)
	ytime_indicator = R_set(ytime)
	ytime_matrix = ytime_indicator - torch.diag(torch.diag(ytime_indicator))
	###T_i is uncensored
	censor_idx = (yevent == 0).nonzero()
	zeros = torch.zeros(n_sample)
	ytime_matrix[censor_idx, :] = zeros
	###1 if pred_i < pred_j; 0.5 if pred_i = pred_j
	pred_matrix = torch.zeros_like(ytime_matrix)
	for j in range(n_sample):
		for i in range(n_sample):
			if pred[i] < pred[j]:
				pred_matrix[j, i]  = 1
			elif pred[i] == pred[j]:
				pred_matrix[j, i] = 0.5

	concord_matrix = pred_matrix.mul(ytime_matrix)
	###numerator
	concord = torch.sum(concord_matrix)
	###denominator
	epsilon = torch.sum(ytime_matrix)
	###c-index = numerator/denominator
	concordance_index = torch.div(concord, epsilon)
	###if gpu is being used
	if torch.cuda.is_available():
		concordance_index = concordance_index.cuda()
	###
	return(concordance_index)


class Cox_PASNet(nn.Module):
	def __init__(self, In_Nodes, Pathway_Nodes, Hidden_Nodes, Out_Nodes, Pathway_Mask, n_clin):
		super(Cox_PASNet, self).__init__()
		self.tanh = nn.Tanh()
		self.pathway_mask = Pathway_Mask
		###gene layer --> pathway layer
		self.sc1 = nn.Linear(In_Nodes, Pathway_Nodes)
		###pathway layer --> hidden layer
		self.sc2 = nn.Linear(Pathway_Nodes, Hidden_Nodes)
		###hidden layer --> hidden layer 2
		self.sc3 = nn.Linear(Hidden_Nodes, Out_Nodes, bias=False)
		###hidden layer 2 + age --> Cox layer
		self.sc4 = nn.Linear(Out_Nodes+n_clin, 1, bias = False)
		self.sc4.weight.data.uniform_(-0.001, 0.001)
		###randomly select a small sub-network
		self.do_m1 = torch.ones(Pathway_Nodes)
		self.do_m2 = torch.ones(Hidden_Nodes)
		###if gpu is being used
		if torch.cuda.is_available():
			self.do_m1 = self.do_m1.cuda()
			self.do_m2 = self.do_m2.cuda()

	def forward(self, x_1, x_2):
		###force the connections between gene layer and pathway layer w.r.t. 'pathway_mask'
		self.sc1.weight.data = self.sc1.weight.data.mul(self.pathway_mask)
		x_1 = self.tanh(self.sc1(x_1))
		if self.training == True: ###construct a small sub-network for training only
			x_1 = x_1.mul(self.do_m1)
		x_1 = self.tanh(self.sc2(x_1))
		if self.training == True: ###construct a small sub-network for training only
			x_1 = x_1.mul(self.do_m2)
		x_1 = self.tanh(self.sc3(x_1))
		###combine age with hidden layer 2
		x_cat = torch.cat((x_1, x_2), 1)
		lin_pred = self.sc4(x_cat)

		return lin_pred


class Cox_PASNet_Model(BaseEstimator, RegressorMixin):
    def __init__(self,
                 pathway_mask,
                 In_Nodes = None,
                 Pathway_Nodes = None,
                 Hidden_Nodes = 100,
                 Out_Nodes = 10,
                 Learning_Rate = 0.01,
                 L2 = 0,
                 Num_Epochs = 100,
                 Dropout_Rate = [0.1, 0.1],
                 clin_covs = None,
                 device='cuda', random_state=123,
                 path = None,
                 refit = False):

        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)

        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted_ = False
        self.training_history_ = {'train_loss': [], 'val_loss': []}

        self.In_Nodes = In_Nodes
        self.Pathway_Nodes = Pathway_Nodes
        self.Hidden_Nodes = Hidden_Nodes
        self.Out_Nodes = Out_Nodes
        self.Learning_Rate = Learning_Rate
        self.L2 = L2
        self.Num_Epochs = Num_Epochs
        self.Dropout_Rate = Dropout_Rate
        self.clin_covs = clin_covs
        self.n_clin = len(self.clin_covs)
        self.pathway_mask = pathway_mask
        self.refit = refit
        self.path = path

        self.training_history_ = {'train_loss': [], 'val_loss': []}

    def fit(self, X, y, patience = 5):
        """
        Trains the Cox-PASNet model using negative partial log-likelihood as the loss function; 
        Adapted to fit into the scikit-learn interface/API.
        
        Args:
            X (pd.DataFrame): Feature matrix (genomic and clinical data).
            y (structured array): Array containing survival time and event status.
            patience (int): Number of epochs to wait for improvement before early stopping.
        
        Returns:
            self: Fitted instance of Cox_PASNet_Model.
        """
        train_x, train_age, train_ytime, train_yevent, \
        eval_x, eval_age, eval_ytime, eval_yevent = self._prepare_data(X, y, val_split = 0.1)
        pathway_mask = self._prepare_pathway(self.pathway_mask, torch.FloatTensor)

        self.model = Cox_PASNet(self.In_Nodes, self.Pathway_Nodes, self.Hidden_Nodes, self.Out_Nodes, pathway_mask, self.n_clin)
        ###if gpu is being used
        if torch.cuda.is_available():
            self.model.cuda()
        ###
        ###optimizer
        opt = optim.Adam(self.model.parameters(), lr=self.Learning_Rate, weight_decay = self.L2)

        counter = 0
        counter = 0
        best_val_loss = float('inf')
        best_model_state = None
        for epoch in range(self.Num_Epochs+1):
            self.model.train()
            opt.zero_grad() ###reset gradients to zeros
            ###Randomize dropout masks
            self.model.do_m1 = dropout_mask(self.Pathway_Nodes, self.Dropout_Rate[0])
            self.model.do_m2 = dropout_mask(self.Hidden_Nodes, self.Dropout_Rate[1])

            pred = self.model(train_x, train_age) ###Forward
            loss = neg_par_log_likelihood(pred, train_ytime, train_yevent) ###calculate loss
            loss.backward() ###calculate gradients
            opt.step() ###update weights and biases

            self.model.sc1.weight.data = self.model.sc1.weight.data.mul(self.model.pathway_mask) ###force the connections between gene layer and pathway layer

            ###obtain the small sub-network's connections
            do_m1_grad = copy.deepcopy(self.model.sc2.weight._grad.data)
            do_m2_grad = copy.deepcopy(self.model.sc3.weight._grad.data)
            do_m1_grad_mask = torch.where(do_m1_grad == 0, do_m1_grad, torch.ones_like(do_m1_grad))
            do_m2_grad_mask = torch.where(do_m2_grad == 0, do_m2_grad, torch.ones_like(do_m2_grad))
            ###copy the weights
            net_sc2_weight = copy.deepcopy(self.model.sc2.weight.data)
            net_sc3_weight = copy.deepcopy(self.model.sc3.weight.data)

            ###serializing net
            net_state_dict = self.model.state_dict()

            ###Sparse Coding
            ###make a copy for net, and then optimize sparsity level via copied net
            copy_net = copy.deepcopy(self.model)
            copy_state_dict = copy_net.state_dict()
            for name, param in copy_state_dict.items():
                ###omit the param if it is not a weight matrix
                if not "weight" in name:
                    continue
                ###omit gene layer
                if "sc1" in name:
                    continue
                ###stop sparse coding
                if "sc4" in name:
                    break
                ###sparse coding between the current two consecutive layers is in the trained small sub-network
                if "sc2" in name:
                    active_param = net_sc2_weight.mul(do_m1_grad_mask)
                if "sc3" in name:
                    active_param = net_sc3_weight.mul(do_m2_grad_mask)
                nonzero_param_1d = active_param[active_param != 0]
                if nonzero_param_1d.size(0) == 0: ###stop sparse coding between the current two consecutive layers if there are no valid weights
                    break
                copy_param_1d = copy.deepcopy(nonzero_param_1d)
                ###set up potential sparsity level in [0, 100)
                S_set =  torch.arange(100, -1, -1)[1:]
                copy_param = copy.deepcopy(active_param)
                S_loss = []
                for S in S_set:
                    param_mask = s_mask(sparse_level = S.item(), param_matrix = copy_param, nonzero_param_1D = copy_param_1d, dtype = dtype)
                    transformed_param = copy_param.mul(param_mask)
                    copy_state_dict[name].copy_(transformed_param)
                    copy_net.train()
                    y_tmp = copy_net(train_x, train_age)
                    loss_tmp = neg_par_log_likelihood(y_tmp, train_ytime, train_yevent).detach().cpu().numpy()
                    S_loss.append(loss_tmp)
                ###apply cubic interpolation
                S_set = S_set.cpu().numpy() if isinstance(S_set, torch.Tensor) else S_set
                S_loss = np.array(S_loss) if isinstance(S_loss, list) else S_loss
                interp_S_loss = interp1d(S_set, S_loss, kind='cubic', axis = 0)
                interp_S_set = torch.linspace(min(S_set), max(S_set), steps=100)
                interp_loss = interp_S_loss(interp_S_set)
                optimal_S = interp_S_set[np.argmin(interp_loss)]
                optimal_param_mask = s_mask(sparse_level = optimal_S.item(), param_matrix = copy_param, nonzero_param_1D = copy_param_1d, dtype = dtype)
                if "sc2" in name:
                    final_optimal_param_mask = torch.where(do_m1_grad_mask == 0, torch.ones_like(do_m1_grad_mask), optimal_param_mask)
                    optimal_transformed_param = net_sc2_weight.mul(final_optimal_param_mask)
                if "sc3" in name:
                    final_optimal_param_mask = torch.where(do_m2_grad_mask == 0, torch.ones_like(do_m2_grad_mask), optimal_param_mask)
                    optimal_transformed_param = net_sc3_weight.mul(final_optimal_param_mask)
                ###update weights in copied net
                copy_state_dict[name].copy_(optimal_transformed_param)
                ###update weights in net
                net_state_dict[name].copy_(optimal_transformed_param)

            self.model.eval()
            eval_pred = self.model(eval_x, eval_age)
            eval_loss = neg_par_log_likelihood(eval_pred, eval_ytime, eval_yevent).view(1,)

            self.training_history_['train_loss'].append(loss)

            self.training_history_['val_loss'].append(eval_loss)
            print("Loss in Eval: ", eval_loss)

            if eval_loss < best_val_loss:
                best_val_loss = eval_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                counter = 0
            else:
                counter += 1

            if counter > patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        self.is_fitted_ = True
        return self

    def _check_early_stopping(self, counter):
        if len(self.training_history_['val_loss']) < 2:
            return 0.0

        if self.training_history_['val_loss'][-1] < self.training_history_['val_loss'][-2]:
            counter = 0.0
        else:
            counter += 1.0
        return counter

    def predict(self, X):
        """
        Predict risk scores for given input data; 
        Adapted to fit into the scikit-learn interface/API.

        Args:
            X (pd.DataFrame): Input feature matrix.

        Returns:
            torch.Tensor: Predicted risk scores (flattened tensor).
        """
        check_is_fitted(self, 'is_fitted_')
        pdata = X.loc[:, self.clin_covs].values
        pdata =  torch.FloatTensor(pdata).to(self.device)
        X = X.drop(self.clin_covs, axis = 1)
        intersect_cols = np.intersect1d(self.pathway_mask.columns, X.columns)
        X = X.loc[: , intersect_cols].values
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            risk_scores = self.model(X, pdata)
        return risk_scores.flatten()

    def score(self, X, y):
        """
        Calculates the C-index for the model's predictions.
        Adapted to fit into the scikit-learn interface/API.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (structured array): Survival time and event status.

        Returns:
            float: C-index.
        """
        check_is_fitted(self, 'is_fitted_')
        preds = self.predict(X)
        event_field = 'status' if 'status' in y.dtype.names else 'event'
        y_time = torch.FloatTensor(y['time'].copy()).to(self.device)
        y_event = np.ascontiguousarray(y[event_field].copy()).astype(np.float32)
        y_event = torch.from_numpy(y_event).to(self.device)
        return c_index(preds, y_time, y_event).cpu().detach().numpy()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def clone(self):
        super(self).clone()

    def _prepare_data(self, X, y, val_split = 0.1):
        """
        Prepares and splits the data into training and validation sets.

        Args:
            X (pd.DataFrame): Feature matrix.
            y (structured array): Array containing survival time and event status.
            val_split (float): Proportion of data to use for validation.

        Returns:
            tuple: Tensors for training and validation sets (features, survival times, event indicators).
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=42)

        event_field_train = 'status' if 'status' in y_train.dtype.names else 'event'
        y_train = pd.DataFrame(y_train).set_index(X_train.index)

        data_train = pd.concat([X_train, y_train], axis = 1, ignore_index=False)
        X_train, times_train, events_train, pdata_train = self._sort_data(data_train, event_field_train, 'time', self.clin_covs)

        X_tensor_train = torch.FloatTensor(X_train).to(self.device)
        time_tensor_train = torch.FloatTensor(times_train).to(self.device)
        event_tensor_train = np.ascontiguousarray(events_train).astype(np.float32)
        event_tensor_train = torch.from_numpy(event_tensor_train).to(self.device)
        pdata_tensor_train = torch.FloatTensor(pdata_train).to(self.device)

        event_field_val = 'status' if 'status' in y_val.dtype.names else 'event'
        y_val = pd.DataFrame(y_val).set_index(X_val.index)
        data_val = pd.concat([X_val, y_val], axis = 1, ignore_index=False)
        X_val, times_val, events_val, pdata_val = self._sort_data(data_val, event_field_val, 'time', self.clin_covs)

        X_tensor_val = torch.FloatTensor(X_val).to(self.device)
        time_tensor_val = torch.FloatTensor(times_val).to(self.device)
        event_tensor_val= np.ascontiguousarray(events_val).astype(np.float32)
        event_tensor_val = torch.FloatTensor(event_tensor_val).to(self.device)
        pdata_tensor_val = torch.FloatTensor(pdata_val).to(self.device)

        return X_tensor_train, pdata_tensor_train, time_tensor_train, event_tensor_train, X_tensor_val, pdata_tensor_val, time_tensor_val, event_tensor_val


    def _sort_data(self, data, event_field, times_field, clin_vars = None):
        """
        Sorts the gene and clinical data based on survival time in descending order.
        (Structure necessary for Cox-PASNet)

        Args:
            data (pd.DataFrame): Combined gene and clinical data.
            event_field (str): Column name for event status.
            times_field (str): Column name for survival time.
            clin_vars (list): List of clinical covariate names.
        
        Returns:
            tuple: Sorted gene data, survival times, event status, and clinical covariates.
        """
        data.sort_values(times_field, ascending = False, inplace = True)
        x = data
        if clin_vars is not None:
            pData = data.loc[:, self.clin_covs].values
            x = data.drop(self.clin_covs, axis = 1)
        x = x.drop([times_field, event_field], axis = 1)
        intersect_cols = np.intersect1d(self.pathway_mask.columns, x.columns)
        x = x.loc[: , intersect_cols].values
        self.In_Nodes = len(intersect_cols)
        ytime = data.loc[:, [times_field]].values
        yevent = data.loc[:, [event_field]].values
        self.pathway_mask = self.pathway_mask.loc[:, intersect_cols]
        self.Pathway_Nodes = self.pathway_mask.shape[0]
        return(x, ytime, yevent, pData)


    def _prepare_pathway(self, pathway_mask, dtype):
        """
        Converts the pathway mask to a PyTorch tensor.

        Args:
            pathway_mask (pd.DataFrame): Bi-adjacency matrix for pathways.
            dtype (torch.dtype): Data type for the tensor.

        Returns:
            torch.Tensor: Tensor representation of the pathway mask.
        """
        pathway_mask = self.pathway_mask.values
        PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
        ###if gpu is being used
        if torch.cuda.is_available():
            PATHWAY_MASK = PATHWAY_MASK.cuda()
        ###
        return(PATHWAY_MASK)

