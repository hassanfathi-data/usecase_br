import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, roc_curve
import plotly.graph_objects as go
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
    
    def compute_feature_importance(self, variables: list, target_var: str = 'd30', test_size: float = 0.2, random_state: int = 42):
        """Test multiple models, select the best one, and return feature importance."""
        model_family = [
            XGBoostModel(random_state=100),
            LightGBMModel(random_state=123),
            LogisticRegressionModel(random_state=456)
        ]
        
        df = self.prepare_features(variables, target_var)
        X = df[variables]
        y = df[target_var]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        
        roc_curves = []
        for model in model_family:
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            model.roc_auc_score = roc_auc
            self.models.append(model)
            
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_curves.append((model.name, fpr, tpr, roc_auc))
        
        self.best_model = max(self.models, key=lambda m: m.roc_auc_score)
        self.best_score = self.best_model.roc_auc_score
        
        fig = go.Figure()
        for name, fpr, tpr, auc in roc_curves:
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'{name} (AUC = {auc:.4f})'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
        fig.update_layout(title=f'ROC Curves Comparison | Best Model: {self.best_model.name} (AUC = {self.best_score:.4f})', 
                         xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=500)
        fig.show()
        
        importance_values = self.best_model.get_feature_importance(variables)
        importance_df = pd.DataFrame({
            'Feature': variables,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False).reset_index(drop=True)
        
        print(f"\nBest Model: {self.best_model.name}")
        print(f"ROC AUC Score: {self.best_score:.4f}")
        
        return importance_df, fig
