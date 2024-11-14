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
        pass
            
    def prepare_data(self, data_config, root): 
        self.dc = DataContainer(data_config=data_config, project_root=root)
        self.X, self.y = self.dc.load_data()
        self.groups = self.dc.get_groups()
    
    # def do_modelling(self, pipeline_steps, config)
    def do_modelling(self, pipeline_steps, param_grid, do_nested_resampling = False, refit = True): 
        if self.X is None or self.y is None: 
            raise Exception("Please call prepare_data() with your preferred config or set X, y, and groups as attributes of your modelling process instance")
        if self.check_pipline_steps:
            self.pipe = Pipeline(pipeline_steps) 
        else: 
            raise Exception("Caution! Your pipline must include a named step for the model of the form ('model', <Instantiated Model Object>)")

        try:
            logger.info("Start model training...")
            logger.info(f"Input data shape: X={self.X.shape}")
                        
            if do_nested_resampling: 
                logger.info("Nested resampling...")
                self.nrs = nested_resampling(self.pipe, self.X, self.y, self.groups, param_grid, self.ss, self.outer_cv, self.inner_cv)
        except Exception as e:
            logger.error(f"Error during nested resampling: {str(e)}")
            raise
        
        if refit: 
            try:
                logger.info("Do HP Tuning for complete model; refit + set complete model")
                self.cmplt_model = self.fit_cmplt_model(param_grid)   
            except Exception as e:
                logger.error(f"Error during final model training: {str(e)}")
                raise
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
    
    