import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import os
import tempfile
from typing import Optional, Sequence
from darts import TimeSeries
from darts.models.forecasting.forecasting_model import ForecastingModel
from huggingface_hub import snapshot_download, upload_folder, create_repo, hf_hub_download
import torch
import tarfile
import os.path
from pathlib import Path

from canswim.paths import symbol_lists_dir


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


class HFHub:
    """
    HuggingFace Hub integration using official HF API.
    https://huggingface.co/docs/huggingface_hub/v0.20.3/en/guides/integrations

    Local-first: dataset/model sync is OFF by default (``hfhub_sync=False``).
    Set ``hfhub_sync=True`` to enable optional remote sync.
    """

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv(override=True)
        if api_key is None:
            api_key = os.getenv("HF_TOKEN")
        self.HF_TOKEN = api_key
        self.data_dir = os.getenv("data_dir", "data")
        self.repo_id = os.getenv("repo_id", "ivelin/canswim")
        # Default False: gather/train/forecast should not block on HF dataset I/O
        self.hfhub_sync = _env_bool("hfhub_sync", default=False)
        logger.info(f"hfhub_sync: {self.hfhub_sync}")
        if self.hfhub_sync and not self.HF_TOKEN:
            raise AssertionError(
                "hfhub_sync is enabled but HF_TOKEN is missing. "
                "Set HF_TOKEN or disable hfhub_sync."
            )

    def upload_model(
        self,
        repo_id: str = None,
        model: ForecastingModel = None,
        private: Optional[bool] = True,
    ):
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping HF model upload.")
            return
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=self.HF_TOKEN,
        )

        with tempfile.TemporaryDirectory() as tmpdirname:
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
            logger.info("Local mode selected. Skipping HF model download.")
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
            logger.info("Local mode selected. Skipping HF timeseries upload.")
            return
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.HF_TOKEN,
        )
        logger.info(f"Uploading timeseries to repo: {repo_id}")
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
            logger.info("Local mode selected. Skipping HF timeseries download.")
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
        """Optional full dataset snapshot (heavy). Disabled unless hfhub_sync=True."""
        if not self.hfhub_sync:
            logger.info(
                "Local mode selected. Skipping HF dataset snapshot_download. "
                "Using local data dir + API refresh only."
            )
            return
        if local_dir is not None:
            data_dir = local_dir
        else:
            data_dir = self.data_dir
        if repo_id is None:
            repo_id = self.repo_id
        Path(data_dir).mkdir(parents=True, exist_ok=True)
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
        forecast_dir = f"{data_dir}/forecast/"
        forecast_tar = f"{data_dir}/forecast.tar.gz"
        if Path(forecast_tar).is_file():
            with tarfile.open(forecast_tar, "r:gz") as tar:
                logger.info(f"Extracting {forecast_tar} to folder {forecast_dir}")
                tar.extractall(path=forecast_dir, filter="data")

    def upload_data(
        self, repo_id: str = None, private: bool = True, local_dir: str = None
    ):
        if not self.hfhub_sync:
            logger.info("Local mode selected. Skipping HF dataset upload.")
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
        repo_info = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
            token=self.HF_TOKEN,
        )
        logger.info(f"repo_info: {repo_info}")
        forecast_dir = f"{data_dir}/forecast/"
        forecast_tar = f"{data_dir}/forecast.tar.gz"
        if Path(forecast_dir).is_dir():
            with tarfile.open(forecast_tar, "w:gz") as tar:
                logger.info(f"Creating {forecast_tar} from folder {forecast_dir}")
                tar.add(forecast_dir, arcname=os.path.basename(forecast_dir))
        logger.info(f"uploading folder {data_dir}")
        upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=data_dir,
            token=self.HF_TOKEN,
        )
        logger.info("Upload finished.")

    def download_symbol_list_csvs(
        self,
        filenames: Optional[Sequence[str]] = None,
        repo_id: Optional[str] = None,
        dest_dir: Optional[Path] = None,
    ) -> Path:
        """Fetch only light CSV symbol lists from the HF dataset (not historical parquet).

        Safe to call even when hfhub_sync is False — this is a one-shot CSV bootstrap.
        Requires HF_TOKEN only if the dataset is private; public files may work without.
        """
        dest = Path(dest_dir) if dest_dir else symbol_lists_dir()
        dest.mkdir(parents=True, exist_ok=True)
        if repo_id is None:
            repo_id = self.repo_id or "ivelin/canswim"
        if filenames is None:
            filenames = [
                "IBD50.csv",
                "IBD250.csv",
                "ibdlive_picks.csv",
                "russell2000_iwm_holdings.csv",
                "sp500_ivv_holdings.csv",
                "nasdaq100_cndx_holdings.csv",
                "watchlist.csv",
                "vti_total_market_stocks.csv",
                "ITB_holdings.csv",
                "IYM_holdings.csv",
                "junk_tickers.csv",
                "ibdfunds.csv",
                "industry_funds.csv",
                "all_stocks.csv",
                "test_stocks.csv",
            ]
        for name in filenames:
            rel = f"data-3rd-party/{name}"
            try:
                path = hf_hub_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    filename=rel,
                    token=self.HF_TOKEN,
                    local_dir=str(dest.parent / "_hf_csv_cache"),
                )
                # Copy/symlink into dest as flat name
                import shutil

                target = dest / name
                shutil.copy2(path, target)
                logger.info(f"Fetched symbol list {name} -> {target}")
            except Exception as e:
                logger.warning(f"Could not download symbol list {rel}: {e}")
        return dest
