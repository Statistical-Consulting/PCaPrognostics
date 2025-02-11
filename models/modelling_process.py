import os
import numpy as np
import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import GridSearchCV
from sklearn.utils import check_random_state
from utils.resampling import nested_resampling
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
    """
    Class to handle the full modelling process for python  models with sklearn-interface. Includes data preparation, cross-validation, 
    hyperparameter tuning, model fitting, and result saving.
    """
    def __init__(self) -> None:
        """
        Initializes the ModellingProcess instance with default attributes for cross-validation, 
        pipeline setup, and data storage.
        """
        self.outer_cv = LeaveOneGroupOut()  # Outer cross-validation strategy: Scikit-learn's LeaveOneGroupOut
        self.inner_cv = LeaveOneGroupOut()  # Inner cross-validation for hyperparameter tuning: Scikit-learn's LeaveOneGroupOut
        self.ss = GridSearchCV  # Searchstrategy: Scikit-learn's GridSearchCV for hyperparameter tuning
        self.pipe = None  # Scikit-learn Pipeline to handle preprocessing and model training
        self.cmplt_model = None  # Final trained model after hyperparameter tuning
        self.cmplt_pipeline = None  # Final pipeline after training
        self.nrs = None  # Nested resampling results
        self.X = None  # Training data features
        self.y = None  # Training data labels
        self.groups = None  # Cohort labels for cross-validation
        self.path = None  # Directory path for saving model and results
        self.fname_cv = None  # File name for saving cross-validation results
        
            
    def prepare_data(self, data_config, root): 
        """
        Prepares training data (group A) by loading features, labels, and groups using a DataContainer instance.
        Sets features, labels, and groups as class instance attributes. 

        Args:
            data_config (dict): Configuration for data loading.
            root (str): Root directory of the project.
        """
        self.dc = DataContainer(data_config=data_config, project_root=root)
        self.X, self.y = self.dc.load_data()
        self.groups = self.dc.get_groups()
    
    def prepare_test_data(self, data_config, root): 
        """
        Prepares independent test data (group B) by loading features, labels, and groups using a DataContainer instance.
        Sets features, labels, and groups as class instance attributes. 

        Args:
            data_config (dict): Configuration for data loading.
            root (str): Root directory of the project.
        """
        self.dc = DataContainer(data_config=data_config, project_root=root)
        self.X_test, self.y_test = self.dc.load_test_data()
        self.test_groups = self.dc.get_test_groups()
    
    def prepare_test_cohort_data(self, data_config, root, cohorts):
        """
        Loads independent test data for specific testing cohorts (group B).

        Args:
            data_config (dict): Configuration for data loading.
            root (str): Root directory of the project.
            cohorts (list): Cohorts of group B.

        Returns:
            tuple: Two lists containing features and targets per cohort.
        """
        dc = DataContainer(data_config=data_config, project_root=root)
        X_cohs = list()
        y_cohs = list()
        print(cohorts)
        if cohorts is not None: 
            for cohort in cohorts: 
                #print(cohort)
                X, y = dc.load_test_data(cohort=cohort)
                X_cohs.append(X)
                y_cohs.append(y)
        return X_cohs, y_cohs
        
    
    def do_modelling(self, pipeline_steps, config): 
        """
        Executes the complete modeling process, including pipeline creation, nested resampling, and final model fitting.

        Args:
            pipeline_steps (list): List of (name, transformer) tuples for creating the pipeline --> objects need to adhere to scikit learn interface /API.
            config (dict): Configuration for the modeling process, including parameters for cross-validation,
                           hyperparameter tuning, and result saving.

        Returns:
            tuple: Nested resampling results, final model, and complete, final pipeline.
        """
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
        """
        Performs hyperparameter tuning and fits the final model on all of group A.

        Args:
            param_grid (dict): Parameter grid for GridSearchCV.
            monitor (optional): Additional monitor object for evaluation during training.

        Returns:
            tuple: The best model and the complete resampling result.
        """
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
        """
        Save the model, cross-validation results, and pipeline to the specified directory.

        Args:
            path (str): Directory path to save the results.
            fname (str): File name for saving the results.
            model (optional): Trained model to save as a pickle file.
            cv_results (optional): Cross-validation results to save as a CSV file.
            pipe (optional): Pipeline to save as a pickle file.

        Returns:
            None
        """
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

    
    def _check_modelling_prerequs(self, pipeline_steps): 
        """
        Checks whether the necessary prerequisites for the modeling process are met (data is prepared + model exists in pipeline).

        Args:
            pipeline_steps (list): List of (name, transformer) tuples representing the steps in the pipeline.

        Returns:
            tuple: A boolean indicating if an error was found (True if an error exists), 
                and a string message explaining the error.
        """
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
        """
        Extracts configuration values from the provided modelling dictionary.

        Args:
            config (dict): Configuration dictionary with keys such as 'params_cv', 'monitor', 
                        'do_nested_resampling', and 'refit'.

        Returns:
            tuple: Contains the following extracted values:
                - param_grid (dict or None): Parameter grid for cross-validation.
                - monitor (object or None): Optional monitor object for early stopping.
                - do_nested_resampling (bool): Indicates whether nested resampling should be performed.
                - refit_hp_tuning (bool): Indicates whether to refit the model with hyperparameter tuning.
        """
        if config.get("params_cv", None) is None: 
            logger.warning("No param grid for (nested) resampling detected - will fit model with default HPs and on complete data")
            return None, False, False
        if config.get('monitor', None) is None: 
            logger.info("No additional monitoring detected")
        return config['params_cv'], config.get('monitor', None), config.get('do_nested_resampling', True) , config.get('refit', True)
    
    def set_params(self, params):
        """
        Set attributes of the current object based on a dictionary of parameters.
        """
        for key, value in params.items():
            setattr(self, key, value) 
    
    def _set_seed(self, seed = 1234):
        """
        Set the random seed for NumPy, PyTorch, and scikit-learn to ensure reproducibility.
        """
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
    