import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from scripts.models.xgboost_model import XGBoostModel
from scripts.models.random_forest_model import RandomForestModel
from scripts.models.logistic_regression_model import LogisticRegressionModel


def calculate_models_performance(models: list, X_train, X_test, y_train, y_test):
    """Train models and calculate their performance metrics."""
    roc_data = []
    trained_models = []
    for model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_pred)
        model.roc_auc_score = auc
        trained_models.append(model)
        fpr, tpr, _ = roc_curve(y_test, y_pred)
        roc_data.append((model.name, fpr, tpr, auc))
    best_model = max(trained_models, key=lambda m: m.roc_auc_score)
    best_score = best_model.roc_auc_score
    return roc_data, best_model, best_score, trained_models


def compute_feature_importance(X_train, X_test, y_train, y_test, variables: list):
    """Train multiple models, select the best one, and return feature importance."""
    models = [XGBoostModel(random_state=100), RandomForestModel(random_state=123), LogisticRegressionModel(random_state=456)]
    roc_data, best_model, best_score, trained_models = calculate_models_performance(models, X_train, X_test, y_train, y_test)
    
    importance_df = pd.DataFrame({'Feature': variables, 'Importance': best_model.get_feature_importance(variables)}).sort_values('Importance', ascending=False).reset_index(drop=True)
    print(f"\nBest Model: {best_model.name}\nROC AUC Score: {best_score:.4f}")
    return importance_df, roc_data, best_model, best_score
