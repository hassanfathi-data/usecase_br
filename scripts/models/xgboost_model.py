from xgboost import XGBClassifier
from .model_parent import ModelParent


class XGBoostModel(ModelParent):
    """XGBoost model class."""
    def __init__(self, random_state: int = 42):
        super().__init__("XGBoost", random_state)
    
    def get_model(self):
        return XGBClassifier(random_state=self.random_state, eval_metric='logloss')

