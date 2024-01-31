import pandas as pd
from dotenv import load_dotenv
import os
import tempfile
from typing import Optional
from darts.models.forecasting.forecasting_model import ForecastingModel
from huggingface_hub import snapshot_download, upload_folder


class HFHub:
    """HuggingFace Hub integration"""

    def __init__(self, api_key: Optional[str] = None):
        print("HFHub init")
        if api_key is None:
            # load from .env file or OS vars if available
            load_dotenv(override=True)
            api_key = os.getenv("HF_HUB_TOKEN")
            assert (
                api_key is not None
            ), "Could not find HF_HUB_TOKEN in OS environment. Cannot interact with HF Hub."
        self.HF_HUB_TOKEN = api_key

    def push_data(df: pd.DataFrame, url: str = None):
        pass

    def upload_model(self, model: ForecastingModel = None, repo_id: str = None):
        with tempfile.TemporaryDirectory() as tmpdirname:
            print("created temporary directory for model", tmpdirname)
            model.save(path=f"{tmpdirname}/{model.model_name}")
            upload_folder(
                repo_id=repo_id, folder_path=tmpdirname, token=self.HF_HUB_TOKEN
            )

    def download_model(
        self,
        repo_id: str = None,
        model_name: str = None,
        model_class: object = None,
    ) -> ForecastingModel:
        with tempfile.TemporaryDirectory() as tmpdirname:
            snapshot_download(
                repo_id=repo_id, local_dir=tmpdirname, token=self.HF_HUB_TOKEN
            )
            model = model_class.load(path=f"{tmpdirname}/{model_name}")
            return model
