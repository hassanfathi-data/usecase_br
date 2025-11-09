from lightgbm import LGBMClassifier
from .model_parent import ModelParent


class LightGBMModel(ModelParent):
    """LightGBM model class."""
    def __init__(self, random_state: int = 42):
        super().__init__("LightGBM", random_state)
    
    def get_model(self):
        return LGBMClassifier(random_state=self.random_state, verbose=-1)

