import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
from scripts.models.xgboost_model import XGBoostModel
from scripts.models.lightgbm_model import LightGBMModel
from scripts.models.logistic_regression_model import LogisticRegressionModel


class FeatureImportance:
    def __init__(self, dataset: pd.DataFrame):
        self.df = dataset.copy()
        self.label_encoders = {}
        self.models = []
        self.best_model = None
        self.best_score = None
        self.feature_names = None
    
    def prepare_features(self, variables: list, target_var: str = 'd30'):
        """Prepare features by encoding categorical variables and selecting columns."""
        df = self.df[variables + [target_var]].copy()
        for var in variables:
            if df[var].dtype == 'object' or pd.api.types.is_categorical_dtype(df[var]):
                if var not in self.label_encoders:
                    self.label_encoders[var] = LabelEncoder()
                df[var] = self.label_encoders[var].fit_transform(df[var].astype(str))
        self.feature_names = variables
        return df
    
    def calculate_models_performance(self, models: list, X_train, X_test, y_train, y_test):
        """Train models and calculate their performance metrics."""
        roc_data = []
        for model in models:
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, y_pred)
            model.roc_auc_score = auc
            self.models.append(model)
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_data.append((model.name, fpr, tpr, auc))
        self.best_model = max(self.models, key=lambda m: m.roc_auc_score)
        self.best_score = self.best_model.roc_auc_score
        return roc_data
    
    def compute_feature_importance(self, variables: list, target_var: str = 'd30', test_size: float = 0.2, random_state: int = 42):
        """Test multiple models, select the best one, and return feature importance."""
        models = [XGBoostModel(random_state=100), LightGBMModel(random_state=123), LogisticRegressionModel(random_state=456)]
        df = self.prepare_features(variables, target_var)
        X_train, X_test, y_train, y_test = train_test_split(df[variables], df[target_var], test_size=test_size, random_state=random_state)
        roc_data = self.calculate_models_performance(models, X_train, X_test, y_train, y_test)
        
        importance_df = pd.DataFrame({'Feature': variables, 'Importance': self.best_model.get_feature_importance(variables)}).sort_values('Importance', ascending=False).reset_index(drop=True)
        print(f"\nBest Model: {self.best_model.name}\nROC AUC Score: {self.best_score:.4f}")
        return importance_df, roc_data
