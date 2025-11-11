import numpy as np


class ModelParent:
    """Base class for all classification models."""
    def __init__(self, name: str, random_state: int = 73):
        self.name = name
        self.random_state = random_state
        self.model = None
        self.roc_auc_score = None
    
    def get_model(self):
        """Return the model instance. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def fit(self, X_train, y_train):
        """Train the model."""
        self.model = self.get_model()
        self.model.fit(X_train, y_train)
    
    def predict_proba(self, X):
        """Get probability predictions."""
        return self.model.predict_proba(X)[:, 1]
    
    def get_feature_importance(self, variables: list):
        """Extract feature importance from the model based on model type."""
        if self.name in ['XGBoost', 'Random Forest']:
            return self.model.feature_importances_
        elif self.name == 'Logistic Regression':
            return np.abs(self.model.coef_[0])
        else:
            return np.zeros(len(variables))

