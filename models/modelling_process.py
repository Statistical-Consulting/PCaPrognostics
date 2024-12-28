import os
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.utils import check_random_state
from utils.resampling import nested_resampling
from utils.visualization import plot_survival_curves, plot_cv_results, plot_feature_importance
from preprocessing.data_container import DataContainer
import pickle
import torch
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModellingProcess(): 
    def __init__(self) -> None:
        self.outer_cv = LeaveOneGroupOut()
        self.inner_cv = LeaveOneGroupOut()
        self.ss = GridSearchCV
        self.pipe = None
        self.cmplt_model = None
        self.cmplt_pipeline = None
        self.nrs = None
        self.X = None
        self.y = None
        self.groups = None
        self.path = None
        self.fname_cv = None
        pass
            
    def prepare_data(self, data_config, root): 
        self.dc = DataContainer(data_config=data_config, project_root=root)
        self.X, self.y = self.dc.load_data()
        self.groups = self.dc.get_groups()
    
    def get_testing_cohorts(self): 
        ind_X, ind_y = self.dc.load_val_cohrts()
        
    def do_external_validation(self, model_path): 
        if self.cmplt_pipeline is None:
            self.load_pipe(model_path)     
        ind_X, ind_y = self.dc.load_val_cohrts()
        
        # TODO: Model/Pipeline.score
    
    def do_modelling(self, pipeline_steps, config): 
        self._set_seed()
        
        if config.get("params_mp", None) is not None: 
            self.set_params(config['params_mp'])
        
        if config.get("path", None) is None or config.get("fname_cv", None) is None: 
            logger.warning("Didn't get sufficient path info for saving cv-results")
        else: 
            self.path = config['path']
            self.fname_cv = config['fname_cv']
        
        err, mes = self._check_modelling_prerequs(pipeline_steps)
        if err: 
           logger.error("Requirements setup error: %s", mes)
           raise Exception(mes)
        else: 
            self.pipe = Pipeline(pipeline_steps) 
        
        param_grid, monitor, do_nested_resampling, refit_hp_tuning = self._get_config_vals(config)
        
        # TODO: do this
        #param_grid = self._prefix_pipeline_params(param_grid, pipeline_steps)
        #print(param_grid)

        try:
            logger.info("Start model training...")
            logger.info(f"Input data shape: X={self.X.shape}")
                        
            if do_nested_resampling: 
                logger.info("Nested resampling...")
                self.nrs = nested_resampling(self.pipe, self.X, self.y, self.groups, param_grid, monitor, self.ss, self.outer_cv, self.inner_cv)
                if (self.fname_cv is not None) and (self.path is not None): 
                    self.save_results(self.path, self.fname_cv, model = None, cv_results = self.nrs, pipe = None)
        except Exception as e:
            logger.error(f"Error during nested resampling: {str(e)}")
            raise
        
        if refit_hp_tuning: 
            try:
                logger.info("Do HP Tuning for complete model; refit + set complete model")
                self.fit_cmplt_model(param_grid)  
                if (self.fname_cv is not None) and (self.path is not None): 
                    self.save_results(self.path, self.fname_cv, model = self.cmplt_model, cv_results = None, pipe = self.cmplt_pipeline) 
            except Exception as e:
                logger.error(f"Error during complete model training: {str(e)}")
                raise    
        elif refit_hp_tuning is False and do_nested_resampling is False: 
            logger.info("Fit complete pipeline wo. HP tuning (on default params)")
            self.cmplt_pipeline = self.pipe.fit(self.X, self.y)
            if (self.fname_cv is not None) and (self.path is not None): 
                    self.save_results(self.path, self.fname_cv, model = None, cv_results = None, pipe = self.cmplt_pipeline)
        
        return self.nrs, self.cmplt_model, self.cmplt_pipeline
    
    
    def fit_cmplt_model(self, param_grid, monitor = None): 
        logger.info("Do HP Tuning for complete model")
        res = self.ss(estimator=self.pipe, param_grid=param_grid, cv=self.outer_cv, n_jobs=4, verbose = 2, refit = True)
        if monitor is not None: 
            res.fit(self.X, self.y, groups = self.groups, model__monitor = monitor)
        else: 
            res.fit(self.X, self.y, groups = self.groups) 
        self.resampling_cmplt = res
        self.cmplt_pipeline = res.best_estimator_
        self.cmplt_model = res.best_estimator_.named_steps['model']
        return res.best_estimator_.named_steps['model'], res  
    
    
    def save_results(self, path, fname, model = None, cv_results = None, pipe = None): 
        """Save model and results"""
        if model is None: 
            logger.warning("Won't save any model, since its not provided")   
        else:  
        # Create directories
            model_dir = os.path.join(path, 'model')
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, f"{fname}.pkl"), 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved model to {model_dir}")
        
        if cv_results is None: 
            logger.warning("Won't save any cv results, since its not provided")
        else: 
            results_dir = os.path.join(path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"{fname}_cv.csv")
            pd.DataFrame(cv_results).to_csv(results_file)
            logger.info(f"Saved CV results to {results_file}")
            
        if pipe is None: 
            logger.warning("Won't save any pipe, since its not provided")
        else:
            pipe_dir = os.path.join(path, 'pipe')
            os.makedirs(pipe_dir, exist_ok=True)
            with open(os.path.join(pipe_dir, f"{fname}.pkl"), 'wb') as f:
                pickle.dump(pipe, f)
            logger.info(f"Saved pipe to {pipe_dir}")



    def save_pipe(self): 
        pass
    
    def load_pipe(self): 
        pass
    
    def load_model(self): 
        pass

    
    def _check_modelling_prerequs(self, pipeline_steps): 
        err = False
        mes = ""
        if self.X is None or self.y is None: 
            mes = mes + "1) Please call prepare_data() with your preferred config or set X, y, and groups as attributes of your modelling process instance"
            err = True
        if not any('model' in tup for tup in pipeline_steps): 
            mes = mes + "2) Caution! Your pipline must include a named step for the model of the form ('model', <Instantiated Model Object>)"
            err = True
        return err, mes

    def _get_config_vals(self, config): 
            if config.get("params_cv", None) is None: 
                logger.warning("No param grid for (nested) resampling detected - will fit model with default HPs and on complete data")
                return None, False, False
            if config.get('monitor', None) is None: 
                logger.info("No additional monitoring detected")
            return config['params_cv'], config.get('monitor', None), config.get('do_nested_resampling', True) , config.get('refit', True)
    
    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value) 
            
            
    def _prefix_pipeline_params(self, params, pipeline_steps):
        """Add pipeline component prefixes to parameters if not already present"""
        prefixed_params = {}
        for param, value in params.items():
            if '__' not in param:
                # Find the relevant step in pipeline_steps
                step_found = False
                for step_name, _ in pipeline_steps:
                    try:
                        # Try setting the parameter to check if it belongs to this step
                        self.model.named_steps[step_name].get_params()[param]
                        prefixed_params[f"{step_name}__{param}"] = value
                        step_found = True
                        break
                    except KeyError:
                        continue
                if not step_found:
                    raise ValueError(f"Could not determine pipeline step for parameter: {param}")
            else:
                prefixed_params[param] = value
        return prefixed_params
    
    def _set_seed(self, seed = 1234):
        np.random.seed(seed)

        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Scikit-learn (and sksurv)
        global random_state
        random_state = check_random_state(seed)
        
        
    def predict_and_plot_survival_function(self, X = None, y = None, estimator = None, n_curves = 5, groups = None):  
        if estimator is None: 
            estimator = self.cmplt_pipeline
        if X is None: 
            X = self.X
        if y is None: 
            y = self.y
        
        plot_survival_curves(estimator, X, y, groups=groups, n_curves=n_curves)
        