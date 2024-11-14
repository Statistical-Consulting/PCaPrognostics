import os
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils.resampling import nested_resampling
from preprocessing.data_container import DataContainer
import pickle

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
        self.nrs = None
        self.X = None
        self.y = None
        self.groups = None
        pass
            
    def prepare_data(self, data_config, root): 
        self.dc = DataContainer(data_config=data_config, project_root=root)
        self.X, self.y = self.dc.load_data()
        self.groups = self.dc.get_groups()
    
    def do_modelling(self, pipeline_steps, config): 
        if config.get("params_mp", None) is not None: 
            self.set_params(config['params_mp'])
        
        err, mes = self.check_modelling_prerequs(pipeline_steps)
        if err: 
           logger.error("Requirements setup error: %s", mes)
           raise Exception(mes)
        else: 
            self.pipe = Pipeline(pipeline_steps) 
        
        param_grid, do_nested_resampling, refit_hp_tuning = self.get_config_vals(config)

        try:
            logger.info("Start model training...")
            logger.info(f"Input data shape: X={self.X.shape}")
                        
            if do_nested_resampling: 
                logger.info("Nested resampling...")
                self.nrs = nested_resampling(self.pipe, self.X, self.y, self.groups, param_grid, self.ss, self.outer_cv, self.inner_cv)
        except Exception as e:
            logger.error(f"Error during nested resampling: {str(e)}")
            raise
        
        if refit_hp_tuning: 
            try:
                logger.info("Do HP Tuning for complete model; refit + set complete model")
                self.cmplt_model = self.fit_cmplt_model(param_grid)   
            except Exception as e:
                logger.error(f"Error during complete model training: {str(e)}")
                raise    
        elif refit_hp_tuning is False and do_nested_resampling is False: 
            logger.info("Fit complete model wo. HP tuning (on default params)")
            self.cmplt_model = self.pipe.fit(self.X, self.y)
        
        return self.nrs
    
    
    def fit_cmplt_model(self, param_grid): 
        logger.info("Do HP Tuning for complete model")
        res = self.ss(estimator=self.pipe, param_grid=param_grid, cv=self.outer_cv, n_jobs=-1, verbose = 2, refit = True)
        res.fit(self.X, self.y, groups = self.groups)
        return res.best_estimator_.named_steps['model']  
    
    
    def save_results(self, path, fname, model = None, cv_results = None,): 
        """Save model and results"""
        if model is None: 
            raise Warning("Won't save any model, since its not provided")   
        else:  
        # Create directories
            model_dir = os.path.join(path, 'model')
            os.makedirs(model_dir, exist_ok=True)
            with open(os.path.join(model_dir, f"{fname}.pkl"), 'wb') as f:
                pickle.dump(self.model, f)
        
        if cv_results is None: 
            raise Warning("Won't save any cv results, since its not provided")
        else: 
            results_dir = os.path.join(path, 'results')
            os.makedirs(results_dir, exist_ok=True)
            results_file = os.path.join(results_dir, f"{fname}_cv_results.csv")
            pd.DataFrame(self.cv_results).to_csv(results_file)
            logger.info(f"Saved CV results to {results_file}")


    def save_pipe(self): 
        pass
    
    def load_pipe(self): 
        pass
    
    def load_model(self): 
        pass
    
    def check_pipline_steps(self, pipeline_steps):
        return any('model' in tup for tup in pipeline_steps)
    
    def check_modelling_prerequs(self, pipeline_steps): 
        err = False
        mes = ""
        if self.X is None or self.y is None: 
            mes = mes + "1) Please call prepare_data() with your preferred config or set X, y, and groups as attributes of your modelling process instance"
            err = True
        if not any('model' in tup for tup in pipeline_steps): 
            mes = mes + "2) Caution! Your pipline must include a named step for the model of the form ('model', <Instantiated Model Object>)"
            err = True
        return err, mes

    def get_config_vals(self, config): 
        if config.get("params_cv", None) is None: 
            logger.warning("No param grid for (nested) resampling detected - will fit model with default HPs and on complete data")
            return None, False, False
        return config['params_cv'], config.get('do_nested_resampling', True) , config.get('refit', True)
    
    def set_params(self, params):
        for key, value in params.items():
            setattr(self, key, value) 
            
            
    def _prefix_pipeline_params(self, params):
        """Add pipeline component prefixes to parameters if not already present"""
        prefixed_params = {}
        for param, value in params.items():
            if '__' not in param:
                # Find the relevant step in pipeline_steps
                step_found = False
                for step_name, _ in self.pipeline_steps:
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