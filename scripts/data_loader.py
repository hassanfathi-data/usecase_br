import pandas as pd
import os

DATA_PATH = "C:/Users/as_cu/Desktop/use_case/ressource"


class DataLoader:
    def __init__(self):
        pass

    def load(self, filename):
        """Load a CSV file from the data path."""
        cwd = os.getcwd()
        file_path = os.path.join(cwd, DATA_PATH, filename)
        data = pd.read_csv(file_path, delimiter=';')
        print(f"Data loaded successfully. Shape: {data.shape}")
        return data

