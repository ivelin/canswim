import pandas as pd
from canswim import ForecastingModelHubMixin
from darts.models import TiDEModel


class TiDEHubModel(TiDEModel, ForecastingModelHubMixin):
    """HF Hub enabled TiDE forecasting model"""

    pass


class HFHub:
    """HuggingFace Hub integration"""

    def __init__(self, api_key=None):
        self.API_KEY = api_key

    def push_data(df: pd.DataFrame, url: str = None):
        pass

    def push_model(local_dir: str = None, remote_path: str = None):
        pass

    def fetch_model(local_dir: str = None, remote_path: str = None):
        pass
