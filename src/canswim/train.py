from canswim.model import CanswimModel
from darts.models import TiDEModel
from canswim.hfhub import HFHub
from darts.metrics import quantile_loss
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
from pandas.tseries.offsets import Day, Week
from loguru import logger


class CanswimTrainer:

    def __init__(self) -> None:
        self.canswim_model = CanswimModel()

    def prepare_data(self):
        self.canswim_model.prepare_data()
        self.canswim_model.plot_splits()
        self.canswim_model.plot_seasonality()

    def build_new_model(self):
        """Build a new model using known optimal hyperparameters"""
        self.canswim_model.build(
            input_chunk_length=252,
            output_chunk_length=42,
            hidden_size=2048,
            num_encoder_layers=3,
            num_decoder_layers=2,
            decoder_output_dim=8,
            temporal_decoder_hidden=80,
            use_layer_norm=True,
            use_reversible_instance_norm=True,
            dropout=0.3,
            optimizer_kwargs={"lr": 1e-05},
            save_checkpoints=False,  # checkpoint to retrieve the best performing model state,
            force_reset=False,
        )

    def plot_backtest_results(self):
        # backtest on 3 stocks
        for i in range(min(len(self.canswim_model.targets_list), 3)):
            start_list = self.canswim_model.get_val_start_list()
            backtest = self.canswim_model.backtest(
                target=self.canswim_model.targets_list[i],
                start=start_list[i],
                past_covariates=self.canswim_model.past_cov_list[i],
                future_covariates=self.canswim_model.future_cov_list[i],
                forecast_horizon=self.canswim_model.pred_horizon,
            )
            # logger.info(f"target series: \n{target}")
            # logger.info(f"backtest series: \n{backtest}")
            loss_vals = []
            for p, b in enumerate(backtest):
                loss = quantile_loss(
                    self.canswim_model.targets_list[i], b, n_jobs=-1, verbose=True
                )
                logger.info(f"quantile loss: {loss} at prediction step {p}")
                loss_vals.append(loss)
            mean_loss = np.mean(loss_vals)
            logger.info(
                f"Mean Backtest Quantile Loss across all prediction periods: {mean_loss}"
            )
            self.canswim_model.plot_backtest_results(
                target=self.canswim_model.targets_list[i],
                backtest=backtest,
                start=start_list[i],
                forecast_horizon=self.canswim_model.pred_horizon,
            )


# main function
def main(new_model: bool = False):

    trainer = CanswimTrainer()

    n_outer_train_loop = 1
    hfhub = HFHub()
    default_repo_id = "ivelin/canswim"
    repo_id = default_repo_id

    def get_env():
        load_dotenv(override=True)
        nonlocal n_outer_train_loop
        n_outer_train_loop = int(os.getenv("n_outer_train_loop", 1))
        nonlocal repo_id
        repo_id = os.getenv("repo_id", default_repo_id)

    get_env()

    logger.info("n_outer_train_loop:", n_outer_train_loop, type(n_outer_train_loop))

    # build a new model or download existing model from hf hub or local dir
    if new_model:
        trainer.build_new_model()
    else:
        trainer.canswim_model.download_model(
            repo_id=repo_id
        )  # prepare next sample subset

    # download market data from hf hub if it hasn't been downloaded already
    trainer.canswim_model.download_data(repo_id=repo_id)

    wall_time = pd.Timestamp.now
    # set to yesterday
    yesterday = wall_time().floor(freq="D") - Day(n=1)
    # set to previous Saturday
    previous_weekend = wall_time().floor(freq="D") - Day(n=wall_time().day_of_week + 2)

    # train loop
    i = 0
    while i < n_outer_train_loop:
        logger.info(f"Outer train loop: {i}")
        try:
            # load a new data sample from local storage
            trainer.canswim_model.load_data()
            # prepare timeseries for training
            trainer.canswim_model.prepare_data()
            # train model
            trainer.canswim_model.train()
        except Exception as e:
            logger.exception(f"Skipping train loop due to ERROR.: {e}")
        # Daily routine
        if wall_time() > yesterday + Day(n=1):
            try:
                logger.info(
                    f"Starting daily routine. Yesterday: {yesterday}, wall_time: {wall_time()}"
                )
                # update yesterday marker to today
                yesterday = wall_time().floor(freq="D")
                # push to hf hub
                trainer.canswim_model.upload_model(repo_id=repo_id)
                # load back latest model weights from hf hub
                trainer.canswim_model.download_model(
                    repo_id=repo_id
                )  # prepare next sample subset
                logger.info("Finished daily routine.")
            except Exception as e:
                logger.exception("ERROR during daily routine: ", e)
        else:
            logger.info(
                f"Skipping daily routine. Yesterday: {yesterday}, wall_time: {wall_time()}"
            )
        # Weekend routine
        if wall_time() > previous_weekend + Week(n=1):
            try:
                logger.info(
                    f"Starting weekend routine. Previous weekend: {previous_weekend}, wall_time: {wall_time()}"
                )
                # update weekend marker to this week's Satuday
                previous_weekend = wall_time().to_period("W").start_time + Day(n=5)
                # gather_market_data() - in range from last gather until now
                # upload_data() - upload changed parquet files
                # run_forecasts() - run model forecast on all stocks which are far enough from previous and next earnings report, save parquet files
                # upload_forecasts() - upload changed parquet files
                logger.info("Finished weekend routine.")
            except Exception as e:
                logger.exception("ERROR during weekend routine: ", e)
        else:
            logger.info(
                f"Skipping weekend routine. Previous weekend: {previous_weekend}, wall_time: {wall_time()}"
            )
        # refresh any config changes in the OS environment
        i += 1
        get_env()
        # Note: load locally saved model
        # since PyTorch Lightning currently does not recommend
        # using fit() multiple times without loading model weights
        # WARNING:darts.models.forecasting.torch_forecasting_model:Attempting to retrain/fine-tune the model without resuming from a checkpoint. This is currently discouraged. Consider model `TiDEModel.load_weights()` to load the weights for fine-tuning.
        trainer.canswim_model.load_model()


if __name__ == "__main__":
    main()
