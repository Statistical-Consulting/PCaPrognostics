from sklearn.feature_selection import SelectFromModel
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, clone, is_classifier
import joblib  
import os


class FoldAwareSelectFromModel(SelectFromModel):
    def __init__(self, estimator, threshold = "mean"):
        self.all_cohorts = ['Atlanta_2014_Long', 'Belfast_2018_Jain', 'CamCap_2016_Ross_Adams',
 'CancerMap_2017_Luca', 'CPC_GENE_2017_Fraser', 'CPGEA_2020_Li',
 'DKFZ_2018_Gerhauser', 'MSKCC_2010_Taylor', 'Stockholm_2016_Ross_Adams']
        super().__init__(estimator=estimator, threshold=threshold)

    
    @_fit_context(
    # SelectFromModel.estimator is not validated yet
    prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None, **fit_params):
        # Example logic to choose a model based on the data split
        root = os.path.dirname(os.getcwd())
        root = os.path.join(root, 'pretrnd_models')
        print(root)
        cohort_names = X.index.to_series().str.split('.').str[0]
        # Get unique cohort names
        unique_cohort_names = cohort_names.unique()
        model_path = ''
        for c in self.all_cohorts: 
            if c not in unique_cohort_names: 
                if len(model_path) > 0: 
                    model_path +=  "_"
                model_path += c  
        if model_path == '': 
            model_path = 'pretrnd_cmplt'
        model_path = os.path.join(root, model_path)
        print(model_path)
        self.estimator= joblib.load(model_path + '.pkl')  
        #super().fit(X, y, **fit_params)  # No need to fit as models are pretrained
        return self