import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from utils.resampling import nested_resampling
from preprocessing.data_container import DataContainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModellingProcess(): 
    def __init__(self) -> None:
        self.outer_cv = LeaveOneGroupOut()
        self.inner_cv = LeaveOneGroupOut()
        self.ss = GridSearchCV()
        self.pipe = None
        self.cmplt_model = None
        pass
            
    def prepare_data(self, data_config, root): 
        self.dc = DataContainer(data_config=data_config, project_root=root)
        self.X, self.y = self.dc.load_data()
        self.groups = self.dc.get_groups()
    
    # def do_modelling(self, pipeline_steps, config)
    def do_modelling(self, pipeline_steps, param_grid, do_nested_resampling = False, refit = True): 
        if self.X is None or self.y is None: 
                raise Exception("Please call prepare_data() with your preferred config or use set_data() to set X, y, and groups")
        
        self.pipe = Pipeline(pipeline_steps) 
             
        ns = None
        res = None
        final_model = None
        try:
            logger.info("Start model training...")
            logger.info(f"Input data shape: X={self.X.shape}")
            
            pipe = Pipeline(pipeline_steps)
            
            if do_nested_resampling: 
                logger.info("Nested resampling...")
                nrs = nested_resampling(pipe, self.X, self.y, self.groups, param_grid, self.inner_cv, self.outer_cv)
        except Exception as e:
            logger.error(f"Error during nested resampling: {str(e)}")
            raise
        
        if refit: 
            try:
                logger.info("Do HP Tuning for complete model; refit + set complete model")
                self.cmplt_model = self.fit_cmplt_model(self, param_grid)   
            except Exception as e:
                logger.error(f"Error during final model training: {str(e)}")
                raise
        return nrs
    
    
    def fit_cmplt_model(self, param_grid): 
        logger.info("Do HP Tuning for complete model")
        res = GridSearchCV(estimator=self.pipe, param_grid=param_grid, cv=self.outer_cv, n_jobs=-1, verbose = 2, refit = True)
        res.fit(self.X, self.y, groups = self.groups)
        return res.best_estimator_.named_steps['model']  
    
    def save_model(): 
        pass

    def save_pipe(): 
        pass
    
    def load_pipe(): 
        pass
    
    def load_model(): 
        pass
    
    def feature_importance(): 
        pass
    