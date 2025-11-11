from sklearn.ensemble import RandomForestClassifier
from .model_parent import ModelParent


class RandomForestModel(ModelParent):
    """Random Forest model class."""
    def __init__(self, random_state: int = 59):
        super().__init__("Random Forest", random_state)
    
    def get_model(self):
        return RandomForestClassifier(random_state=self.random_state, n_estimators=100)

