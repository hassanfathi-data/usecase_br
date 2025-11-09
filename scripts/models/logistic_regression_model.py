from sklearn.linear_model import LogisticRegression
from .model_parent import ModelParent


class LogisticRegressionModel(ModelParent):
    """Logistic Regression model class."""
    def __init__(self, random_state: int = 42):
        super().__init__("Logistic Regression", random_state)
    
    def get_model(self):
        return LogisticRegression(random_state=self.random_state, max_iter=1000)

