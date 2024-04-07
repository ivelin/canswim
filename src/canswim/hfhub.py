import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import os
import tempfile
from typing import Optional
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from huggingface_hub import snapshot_download, upload_folder, create_repo
import torch
import tarfile
import os.path


class HFHub:
    """
    HuggingFace Hub integration using official HF API.
    https://huggingface.co/docs/huggingface_hub/v0.20.3/en/guides/integrations
    """

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv(override=True)
        if api_key is None:
            # load from .env file or OS vars if available
            api_key = os.getenv("HF_TOKEN")
            assert (
                api_key is not None
            ), "Could not find HF_TOKEN in OS environment. Cannot interact with HF Hub."
        self.HF_TOKEN = api_key
        self.data_dir = os.getenv("data_dir", "data")
        self.repo_id = os.getenv("repo_id")
        lm = os.getenv("hfhub_sync", False)
        if isinstance(lm, str) and lm == "False":
            lm = False
        else:
            lm = True
        self.hfhub_sync = lm
        logger.info(f"hfhub_sync: {self.hfhub_sync}")

    def upload_model(
        self,
        repo_id: str = None,
        model: ForecastingModel = None,
        private: Optional[bool] = True,
    ):
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping download.")
            return
        # Create repo if not existing yet and get the associated repo_id
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=self.HF_TOKEN,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
            # logger.info("created temporary directory for model", tmpdirname)
            model.save(path=f"{tmpdirname}/{model.model_name}")
            logger.info(f"Uploading model to repo: {repo_id}")
            upload_folder(repo_id=repo_id, folder_path=tmpdirname, token=self.HF_TOKEN)

    def download_model(
        self,
        repo_id: str = None,
        model_name: str = None,
        model_class: object = None,
        **kwargs,
    ) -> ForecastingModel:
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping download.")
            return
        if torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"
        if repo_id is None:
            repo_id = self.repo_id
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.info(f"Downloading model from repo: {repo_id}")
            snapshot_download(
                repo_id=repo_id, local_dir=tmpdirname, token=self.HF_TOKEN
            )
            logger.info(f"dir file list:\n {os.listdir(tmpdirname)}")
            model = model_class.load(
                path=f"{tmpdirname}/{model_name}", map_location=map_location, **kwargs
            )
            logger.info("Downloaded model from:", repo_id)
            logger.info("Model name:", model.model_name)
            logger.info("Model params:", model.model_params)
            logger.info("Model created:", model.model_created)
            return model

    def upload_timeseries(
        self,
        repo_id: str = None,
        series: TimeSeries = None,
        series_name: str = None,
        private: Optional[bool] = True,
    ):
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping download.")
            return
        # Create repo if not existing
        repo_info = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.HF_TOKEN,
        )
        # logger.info(f"repo_info: ", repo_info)
        logger.info(f"Uploading timesires to repo: {repo_id}")
        df = series.pd_dataframe()
        with tempfile.TemporaryDirectory() as tmpdirname:
            df.to_parquet(path=f"{tmpdirname}/{series_name}.parquet")
            upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=tmpdirname,
                token=self.HF_TOKEN,
            )

    def download_timeseries(
        self,
        repo_id: str = None,
        series_name: str = None,
    ) -> TimeSeries:
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping download.")
            return
        with tempfile.TemporaryDirectory() as tmpdirname:
            logger.info(f"Downloading data from repo: {repo_id}")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=tmpdirname,
                token=self.HF_TOKEN,
            )
            logger.info(os.listdir(tmpdirname))
            df = pd.read_parquet(
                f"{tmpdirname}/{series_name}.parquet", engine="pyarrow"
            )
            ts = TimeSeries.from_dataframe(df)
            return ts

    def download_data(self, repo_id: str = None, local_dir: str = None):
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping download.")
            return
        if local_dir is not None:
            data_dir = local_dir
        else:
            data_dir = self.data_dir
        if repo_id is None:
            repo_id = self.repo_id
        logger.info(
            f"Downloading hf data from {repo_id} to data dir:\n",
            os.listdir(data_dir),
        )
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=data_dir,
            token=self.HF_TOKEN,
        )
        # Unpack forecast parquet files from tar
        forecast_dir = f"{data_dir}/forecast/"
        forecast_tar = f"{data_dir}/forecast.tar.gz"
        with tarfile.open(forecast_tar, "r:gz") as tar:
            logger.info(f"Extracting {forecast_tar} to folder {forecast_dir}")
            tar.extractall(path=forecast_dir, filter="data")

    def upload_data(
        self, repo_id: str = None, private: bool = True, local_dir: str = None
    ):
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping upload.")
            return
        if local_dir is not None:
            data_dir = local_dir
        else:
            data_dir = self.data_dir
        if repo_id is None:
            repo_id = self.repo_id
        logger.info(
            f"Uploading data from local directory: '{data_dir}' to repo id: '{repo_id}'"
        )
        ## Upload all gathered data from 3rd party sources to hf hub
        # prefix for HF Hub dataset repo
        # Create repo if not existing yet
        repo_info = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.HF_TOKEN,
        )
        logger.info(
            f"repo_info: {repo_info}",
        )
        # Compress forecast parquet files to pass hfhub limitation of 25k LFS files
        forecast_dir = f"{data_dir}/forecast/"
        forecast_tar = f"{data_dir}/forecast.tar.gz"
        with tarfile.open(forecast_tar, "w:gz") as tar:
            logger.info(f"Creating {forecast_tar} from folder {forecast_dir}")
            tar.add(forecast_dir, arcname=os.path.basename(forecast_dir))
        # upload select files  to hfhub
        logger.info(f"uploading folder {data_dir}")
        upload_folder(
            repo_id=repo_id,
            # path_in_repo="data-3rd-party",
            repo_type="dataset",
            folder_path=data_dir,
            token=self.HF_TOKEN,
            # ignore_patterns=[forecast_dir],
            # allow_patterns="",
            # delete_patterns=[forecast_dir],
        )
        logger.info(
            "Upload finished.",
        )
