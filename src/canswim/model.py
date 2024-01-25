from pandas.tseries.offsets import BDay
import matplotlib.pyplot as plt
from darts.utils.statistics import plot_acf
from darts.models import TiDEModel
from darts.utils.likelihood_models import QuantileRegression
from darts.models.forecasting.forecasting_model import GlobalForecastingModel
import torch
import random
from canswim.targets import Targets
from canswim.covariates import Covariates
from dotenv import load_dotenv
import pandas as pd
import os
from darts.metrics import rmsle
from darts import TimeSeries


class CanswimModel:
    def __init__(self):
        self.n_stocks: int = 50
        self.n_epochs: int = 10
        self.train_series = {}
        self.val_series = {}
        self.test_series = {}
        self.past_covariates_train = {}
        self.past_covariates_val = {}
        self.past_covariates_test = {}
        self.val_start = {}
        self.test_start = {}
        self.torch_model: TiDEModel = None
        self.saved_model_name = "data/canswim_model.pt"
        self.n_plot_samples: int = 4
        self.train_date_start: pd.Timestamp = None
        # How far back should the model look in order to make a prediction
        self.train_history: int = 252 * 2  # 252 days in a year with market data
        # How far into the future should the model forecast
        self.pred_horizon: int = 21 * 2  # 21 days in a month with market data

        self.n_test_range_days: int = self.train_history + 3 * self.pred_horizon
        print(f"n_test_range_days: {self.n_test_range_days}")

        # minimum amount of historical data required to train on a stock series
        # stocks that are too new off IPOs, are not a good fit for training this model
        self.min_samples = self.n_test_range_days * 3
        print(f"min_samples: {self.min_samples}")

        self.targets = Targets(min_samples=self.min_samples)
        self.covariates = Covariates()

        # load from .env file or OS vars if available
        load_dotenv(override=True)

        self.n_stocks = int(
            os.getenv("n_stocks", 50)
        )  # -1 for all, otherwise a number like 300
        print("n_stocks: ", self.n_stocks)
        self.n_epochs = int(os.getenv("n_epochs", 5))  # model training epochs
        print("n_epochs: ", self.n_epochs)

        # pick the earlies date after which market data is available for all covariate series
        self.train_date_start = pd.Timestamp(
            os.getenv("train_date_start", "1991-01-01")
        )
        # use GPU if available
        if torch.cuda.is_available():
            print("Configuring CUDA GPU")
            # utilize CUDA tensor cores with bfloat16
            # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
            torch.set_float32_matmul_precision("high")  #  | 'medium'

    def align_targets_and_covariates():
        # focus on tickers that we have full data sets for training and validation
        target_set = set(target_series.keys())
        future_set = set(future_covariates.keys())
        past_set = set(past_covariates.keys())
        tickers_with_complete_data = target_set & future_set & past_set
        tickers_without_complete_data = (
            target_set | future_set | past_set
        ) - tickers_with_complete_data
        print(
            f"Removing time series for tickers with incomplete data sets: {tickers_without_complete_data}. Keeping {tickers_with_complete_data} "
        )

        new_target_series = {}
        new_future_covariates = {}
        new_past_covariates = {}

        for t in tickers_with_complete_data:
            new_target_series[t] = target_series[t]
            new_future_covariates[t] = future_covariates[t]
            new_past_covariates[t] = past_covariates[t]

        target_series = new_target_series
        future_covariates = new_future_covariates
        past_covariates = new_past_covariates

        # align targets with past covs
        for t, covs in past_covariates.items():
            ts_sliced = target_series[t].slice_intersect(covs)
            target_series[t] = ts_sliced
            covs_sliced = covs.slice_intersect(ts_sliced)
            past_covariates[t] = covs_sliced

        # align targets with future covs
        for t, covs in future_covariates.items():
            ts_sliced = target_series[t].slice_intersect(covs)
            target_series[t] = ts_sliced
            covs_sliced = covs.slice_intersect(ts_sliced)
            future_covariates[t] = covs_sliced

    def __prepare_data_splits(self):
        for t, target in self.target_series.items():
            self.test_start[t] = target.end_time() - BDay(n=self.n_test_range_days)
            self.val_start[t] = self.test_start[t] - BDay(n=self.n_test_range_days)
        for t, target in self.target_series.items():
            if len(target) > self.min_samples:
                print(f"preparing train, val split for {t}")
                print(f"{t} test_start: {self.test_start[t]}")
                print(f"{t} val_start: {self.val_start[t]}")
                print(
                    f"{t} start time, end time: {target.start_time()}, {target.end_time()}"
                )
                train, val = target.split_before(self.val_start[t])
                val, test = val.split_before(self.test_start[t])
                # there should be no gaps in the training data
                assert len(train.gaps().index) == 0
                assert (
                    len(val) >= self.n_test_range_days
                ), f"val samples {len(val)} but must be at least {n_test_range_days}"
                assert (
                    len(test) >= n_test_range_days
                ), f"test samples {len(test)} but must be at least {n_test_range_days}"
                self.train_series[t] = train
                self.val_series[t] = val
                self.test_series[t] = test
                past_train, past_val = self.past_covariates[t].split_before(
                    self.val_start[t]
                )
                past_val, past_test = past_val.split_before(self.test_start[t])
                # there should be no gaps in the training data
                assert len(past_train.gaps().index) == 0
                self.past_covariates_train[t] = past_train
                self.past_covariates_val[t] = past_val
                self.past_covariates_test[t] = past_test
            else:
                print(
                    f"Removing {t} from train set. Not enough samples. Minimum {self.min_samples} needed, but only {len(target)} available"
                )
        self.targets_list = [
            series for ticker, series in sorted(self.target_series.items())
        ]
        self.target_train_list = [
            series for ticker, series in sorted(self.train_series.items())
        ]
        # print(len(self.target_train_list))
        self.target_val_list = [
            series for ticker, series in sorted(self.val_series.items())
        ]
        # print(len(target_val_list))
        self.target_test_list = [
            series for ticker, series in sorted(self.test_series.items())
        ]
        # print(len(target_test_list))
        self.past_cov_list = [
            series for ticker, series in sorted(self.past_covariates_train.items())
        ]
        # print(len(past_cov_list))
        self.past_cov_val_list = [
            series for ticker, series in sorted(self.past_covariates_val.items())
        ]
        # print(len(past_cov_val_list))
        self.past_cov_test_list = [
            series for ticker, series in sorted(self.past_covariates_test.items())
        ]
        # print(len(past_cov_test_list))
        self.future_cov_list = [
            series for ticker, series in sorted(self.future_covariates.items())
        ]
        # print(len(future_cov_list))

    def __drop_incomplete_series(self):
        # remove tickers that don't have sufficient training data
        target_series_ok = {}
        for t, target in self.target_series.items():
            # only include tickers with sufficient data for training
            if len(target) >= self.min_samples:
                target_series_ok[t] = target
            else:
                del self.past_covariates[t]
                del self.future_covariates[t]
                # print(f'preparing train, val split for {t}')
                print(f"Removing {t} from training loop. Not enough samples")
        return target_series_ok

    @property
    def stock_tickers(self):
        return self.__stock_tickers

    def __prepare_stock_price_series(self, tickers: set = None, train_date_start=None):
        print(f"Preparing ticker series for {len(tickers)} stocks.")
        stock_price_series = {
            t: TimeSeries.from_dataframe(self.stock_price_dict[t], freq="B")
            for t in tickers
        }
        print("Ticker series dict created.")
        filler = MissingValuesFiller()
        for t, series in stock_price_series.items():
            # gaps = series.gaps(mode="any")
            # print(f'ticker: {t} gaps: \n {gaps}')
            series_filled = filler.transform(series)
            # check for any data gaps
            any_price_gaps = series_filled.gaps(mode="any")
            assert not any_price_gaps
            # print(f'ticker: {t} gaps after filler: \n {any_price_gaps}')
            stock_price_series[t] = series_filled
        print("Filled missing values in ticker series.")
        for t, series in stock_price_series.items():
            stock_price_series[t] = series.slice(train_date_start, series.end_time())
            # print(f'ticker: {t} , {ticker_series[t]}')
        # add holidays as future covariates
        print("Aligned ticker series dict with train start date.")
        print("Ticker series prepared.")
        return stock_price_series

    def prepare_data(self):
        # reduce ticker set to a workable sample size for one training loop
        self.__stock_tickers = random.sample(
            self.targets.all_stock_tickers, self.n_stocks
        )
        print("Training loop stock subset:", self.stock_tickers)

        # prepare stock price time series
        ## ticker_train_dict = dict((k, self.ticker_dict[k]) for k in self.stock_tickers)
        self.stock_price_series = self.__prepare_stock_price_series(
            tickers=self.stock_tickers, train_date_start=self.train_date_start
        )
        # prepare target time series
        target_columns = ["Close"]
        self.target_series = self.targets.prepare_target_series(
            stock_price_series=self.stock_price_series, target_columns=target_columns
        )
        # prepare past covariates
        self.past_covariates = self.covariates.prepare_past_covariates(
            stock_price_series=self.stock_price_series, target_columns=target_columns
        )
        # prepare future covariates
        self.future_covariates = self.covariates.prepare_future_covariates(
            ticker_series=self.ticker_series, min_samples=self.min_samples
        )
        target_series_ok = self.__drop_incomplete_series()
        target_series = target_series_ok
        print(f"Preparing train, val, test splits")
        self.__prepare_data_splits()

    def load_model(self):
        if torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"
        try:
            self.torch_model = TiDEModel.load(
                self.saved_model_name, map_location=map_location
            )
            return True
        except Exception as e:
            print("Unable to find or load a saved model. Error: \n", e)
        return False

    def __merge_model_params(self):
        pass

    def build_model(self, hparams=None):
        # scaler = Scaler(verbose=True, n_jobs=-1)
        # darts encoder examples: https://unit8co.github.io/darts/generated_api/darts.dataprocessing.encoders.encoders.html#
        # Prepare Encoders that Darts will automatically use for training and inference
        encoders = {
            "cyclic": {"future": ["dayofweek", "month", "quarter"]},
            "datetime_attribute": {"future": ["dayofweek", "month", "quarter", "year"]},
            "position": {"past": ["relative"], "future": ["relative"]},
            "custom": {
                "future": [lambda idx: (idx.year % 4)]
            },  # signal proximity to US election years, which is known to have significance to market cycles.
            # "transformer": scaler
        }
        # hyperparamter selection
        # based on Darts template: https://unit8co.github.io/darts/examples/18-TiDE-examples.html#Model-Parameter-Setup
        optimizer_kwargs = {
            "lr": 2.24e-4,
        }
        # PyTorch Lightning Trainer arguments
        pl_trainer_kwargs = {
            "gradient_clip_val": 1,
            "max_epochs": 200,
            # "accelerator": "auto",
            "accelerator": "gpu",
            "accelerator": "gpu",
            "devices": [0],
            # "auto_select_gpus": True,
            "callbacks": [],
        }
        # pl_trainer_kwargs = {"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
        # learning rate scheduler
        lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
        lr_scheduler_kwargs = {
            "gamma": 0.999,
        }
        # early stopping (needs to be reset for each model later on)
        # this setting stops training once the the validation loss has not decreased by more than 1e-3 for 10 epochs
        early_stopping_args = {
            "monitor": "val_loss",
            "patience": 10,
            "min_delta": 1e-3,
            "mode": "min",
        }
        #
        common_model_args = {
            # "input_chunk_length": 12,  # lookback window
            # "output_chunk_length": 12,  # forecast/lookahead window
            "optimizer_kwargs": optimizer_kwargs,
            "pl_trainer_kwargs": pl_trainer_kwargs,
            "lr_scheduler_cls": lr_scheduler_cls,
            "lr_scheduler_kwargs": lr_scheduler_kwargs,
            # "likelihood": None,  # use a likelihood for probabilistic forecasts
            # "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
            # "force_reset": True,
            "batch_size": 256,  # 512,
            "random_state": 42,
        }
        model = None
        print("Creating a new model")
        # using TiDE hyperparameters from Table 8 in section B.3 of the original paper
        # https://arxiv.org/pdf/2304.08424.pdf
        model_params = self.__merge_model_params(
            hparams=hparams,
            **common_model_args,
            input_chunk_length=self.train_history,
            output_chunk_length=self.pred_horizon,
            add_encoders=None,  # encoders,
            hidden_size=256,  # 512,
            num_encoder_layers=2,
            num_decoder_layers=2,
            decoder_output_dim=8,  # 32,
            temporal_decoder_hidden=16,  # 64, # 128,
            dropout=0.2,
            use_layer_norm=True,
            use_reversible_instance_norm=True,
            n_epochs=self.n_epochs,
            likelihood=QuantileRegression(
                quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]
            ),
            # model_name=self.saved_model_name,
            log_tensorboard=True,
            nr_epochs_val_period=1,
        )
        model = TiDEModel(model_params)
        self.torch_model = model

    def train(self):
        # load or create a new model instance
        self.torch_model = self.__load_model()
        # when True, multiple time series are supported
        supports_multi_ts = issubclass(
            self.torch_model.__class__, GlobalForecastingModel
        )
        assert supports_multi_ts is True
        # train model
        # for i in range(100):
        self.torch_model.fit(
            self.target_train_list,
            epochs=self.n_epochs,
            past_covariates=self.past_cov_list,
            future_covariates=self.future_cov_list,
            val_series=self.target_val_list,
            val_past_covariates=self.past_cov_val_list,
            val_future_covariates=self.future_cov_list,
            verbose=True,
            num_loader_workers=4,  # num_loader_workers recommended at 4*n_GPUs
        )

    def plot_splits(self):
        # plot sample of target series
        fig, axes = plt.subplots(nrows=self.n_plot_samples, ncols=1, figsize=(20, 20))
        axes2 = {}
        for i in range(self.n_plot_samples):
            axes2[i] = axes[i].twinx()
        for i, t in enumerate(self.target_series.keys()):
            if i > self.n_plot_samples - 1:
                break
            self.train_series[t]["Close"].plot(
                label=f"ticker {t} Close train", ax=axes[i]
            )
            #    train_series[t]['Volume'].plot(label=f'ticker {t} Volume train', ax=axes2[i])
            self.val_series[t]["Close"].plot(label=f"ticker {t} Close val", ax=axes[i])
            #    val_series[t]['Volume'].plot(label=f'ticker {t} Volume val', ax=axes2[i])
            self.test_series[t]["Close"].plot(
                label=f"ticker {t} Close test", ax=axes[i]
            )
        #    test_series[t]['Volume'].plot(label=f'ticker {t} Volume test', ax=axes2[i])
        axes[0].set_ylabel("Target Series")

    def plot_seasonality(self):
        # plot sample of target series
        fig, axes = plt.subplots(nrows=1, ncols=self.n_plot_samples, figsize=(12, 4))
        for i, t in enumerate(self.target_series.keys()):
            if i >= self.n_plot_samples:
                break
            plot_acf(self.train_series[t]["Close"], alpha=0.05, axis=axes[i])

        axes[0].set_ylabel("Seasonality")

    def find_model(self):
        # hparams_grid = {
        #      **common_model_args,
        #      'input_chunk_length': train_history,
        #      'output_chunk_length': pred_horizon,
        #      'add_encoders': None, # encoders,
        #      'hidden_size': [256, 512, 1024],
        #      'num_encoder_layers': [1, 2, 3],
        #      'num_decoder_layers': [1, 2, 3],
        #      'decoder_output_dim': [4, 8, 16, 32],
        #      'temporal_decoder_hidden': [16, 32, 64, 128],
        #      'dropout': [0.0, 0.1, 0.2, 0.3, 0.5],
        #      'use_layer_norm': [True, False],
        #      'lr': [1e-1, 1e-2, 1-3, 1-4, 1e-5],
        #      'use_reversible_instance_norm': [True, False],
        #      'n_epochs': n_epochs,
        #      'likelihood': QuantileRegression(quantiles=[0.01, 0.05, 0.2, 0.5, 0.8, 0.95, 0.99]),
        #      # model_name: saved_model_name,
        #      'log_tensorboard': True,
        #      'nr_epochs_val_period':1
        # }
        # best_model, best_hparams, lerror = TiDEModel.gridsearch(parameters=hparams_grid,
        #                                                        series=target_train_list[0],
        #                                                        past_covariates=past_cov_list[0],
        #                                                        future_covariates=future_cov_list[0],
        #                                                        stride=1,
        #                                                        start=0.5,
        #                                                        start_format='value',
        #                                                        last_points_only=False,
        #                                                        val_series=target_val_list[0],
        #                                                        verbose=True,
        #                                                        n_jobs=-1,
        #                                                    )
        self.torch_model = optimal_model

    def save(self):
        self.torch_model.save(self.saved_model_name)

    def load_data(self):
        self.targets.load_data()
        self.covariates.load_data()

    def __get_val_start_list():
        val_start_list = []
        for t, target in sorted(target_series.items()):
            val_start_list.append(val_start[t])
        return val_start_list

    def __get_test_start_list():
        val_test_list = []
        for t, target in sorted(target_series.items()):
            val_test_list.append(test_start[t])
        return val_test_list

    def __get_pred_start(start_times=None, offset=None):
        pred_start = {}
        bdays_offset = BDay(n=train_history + offset * pred_horizon)
        for t, start_time in start_times.items():
            pred_start[t] = start_time + bdays_offset
        return pred_start

    def __get_pred_list(pred_start=None):
        pred_list = []
        for t, target in sorted(target_series.items()):
            pred_series = target.slice(
                target.start_time(), pred_start[t] - pd.Timedelta(days=1)
            )
            pred_list.append(pred_series)
        return pred_list

    def __get_past_cov_list(pred_start=None):
        past_cov_list = []
        for t, past_cov in sorted(past_covariates.items()):
            past_covs_sliced = past_cov.slice(
                past_cov.start_time(), pred_start[t] - pd.Timedelta(days=1)
            )
            past_cov_list.append(past_covs_sliced)
        return past_cov_list

    def __get_pred(pred_list=None, past_cov_list=None):
        # pred2 = model.predict(pred_horizon, series=pred2_series, past_covariates=past_covariates, future_covariates=future_covariates, mc_dropout=True, num_samples=500) #   len(val))
        pred = model.predict(
            pred_horizon,
            series=pred_list,
            mc_dropout=True,
            num_samples=500,
            past_covariates=past_cov_list,
            future_covariates=future_cov_list,
            num_loader_workers=4,
        )
        return pred

    def test(self):
        pred_test_outputs = []
        # get predictions at several points in time over the validation set
        for w in range(3):
            pred_start = self.__get_pred_start(start_times=test_start, offset=w)
            pred_list = self.__get_pred_list(pred_start)
            past_cov_list = self.__get_past_cov_list(pred_start)
            # print(f'pred_list: \n{pred_list}')
            pred = self.__get_pred(pred_list=pred_list, past_cov_list=past_cov_list)
            pred_test_outputs.append(pred)

        pred_val_outputs = []
        # get predictions at several points in time over the validation set
        for w in range(15):
            pred_start = get_pred_start(start_times=val_start, offset=w)
            pred_list = get_pred_list(pred_start)
            past_cov_list = get_past_cov_list(pred_start)
            # print(f'pred_list: \n{pred_list}')
            pred = get_pred(pred_list=pred_list, past_cov_list=past_cov_list)
            pred_val_outputs.append(pred)

    def plot_test_results(self):
        # select a reasonable range of train and val data points for convenient visualization of results
        actual = {}

        def plot_actual():
            for i, t in enumerate(sorted(train_series.keys())):
                if i < n_plot_samples:
                    tsliced = target_series[t]
                    actual[t] = tsliced.slice(val_start[t], target.end_time())
                    # ax = actual[t]['Open'].plot(label='actual Open', linewidth=1, ax=axes[i])
                    ax = actual[t]["Close"].plot(
                        label="actual Close", linewidth=1, ax=axes[i]
                    )
                    vol = past_covariates[t]["Volume"].slice(
                        val_start[t], target.end_time()
                    )
                    vol.plot(label="actual Volume", linewidth=1, ax=axes2[i])

        def plot_pred(pred_out=None, past_cov_list=None):
            # pred2 = model.predict(pred_horizon, series=pred2_series, past_covariates=past_covariates, future_covariates=future_covariates, mc_dropout=True, num_samples=500) #   len(val))
            for i, t in enumerate(sorted(train_series.keys())):
                if i < n_plot_samples:
                    # ax = pred_out[i]['Open'].plot(label=f'forecast Open', linewidth=2, ax=axes[i])
                    ax = pred_out[i]["Close"].plot(
                        label=f"forecast Close", linewidth=2, ax=axes[i]
                    )
                    plt.legend()
                    # Major ticks every half year, minor ticks every month,
                    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(13)))
                    ax.xaxis.set_minor_locator(mdates.MonthLocator())
                    ax.grid(True)
                    ax.set_ylabel(f"{t}")

        fig, axes = plt.subplots(nrows=n_plot_samples, ncols=1, figsize=(20, 12))
        axes2 = {}
        for i in range(n_plot_samples):
            axes2[i] = axes[i].twinx()

        plot_actual()

        # plot predictions at several points in time over the validation set
        # plot_pred(pred_outputs=pred_outputs, past_cov_list=past_cov_list)
        for pred_out in pred_val_outputs:
            plot_pred(pred_out=pred_out, past_cov_list=past_cov_list)

        for pred_out in pred_test_outputs:
            plot_pred(pred_out=pred_out, past_cov_list=past_cov_list)

    def backtest(
        self,
        series=None,
        start=None,
        past_covariates=None,
        future_covariates=None,
        forecast_horizon=None,
    ):
        """Backtest model on the full range of test data"""
        # set the forecast start at a time before the validation date in order to see
        # the difference between predicting on training vs validation data
        # predicting up to the validate date should be near match to actuals
        # whereas predicting on unseen validate data should have room for improvement
        # forecast_start = test_start-BDay(n=30)
        pred_start_list = self.__get_test_start_list()
        forecast_start = pred_start_list[0] - BDay(n=30)
        forecast_horizon = self.pred_horizon  # pred_horizon
        # Past and future covariates are optional because they won't always be used in our tests
        # We backtest the model on the val portion of the flow series, with a forecast_horizon:
        backtest = self.torch_model.historical_forecasts(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            start=start,
            retrain=False,
            verbose=True,
            forecast_horizon=forecast_horizon,
            overlap_end=True,
            stride=forecast_horizon,
            last_points_only=False,
            num_samples=500,  # probabilistic forecasting
            predict_kwargs={"mc_dropout": True, "num_loader_workers": 4, "n_jobs": -1},
        )
        # use Log RMSE to remove the big variance in absolute price values between difference stocks
        test_error = rmsle(self.targets_list[0], backtest[0])
        # print(f"Backtest RMSLE = {test_error}")
        return backtest, test_error

    def plot_backtest_results(self, backtest):
        fig, axes = plt.subplots(figsize=(20, 10))
        # axes2 = axes.twinx()

        actual_sliced = targets_list[0].slice(
            forecast_start - pd.Timedelta(days=train_history), target.end_time()
        )
        ax = actual_sliced["Close"].plot(label="actual Close", linewidth=2, ax=axes)
        # past_cov_list[i]['Volume'].plot(label='actual Volume', ax=axes2)

        # Major ticks every half year, minor ticks every month,
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(13), interval=1))
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.grid(True)
        ax.set_ylabel(f"{t}")

        for i, b in enumerate(backtest):
            # if i < n_plot_samples:
            b.plot(
                label=f"backtest Close (forecast_horizon={forecast_horizon})",
                linewidth=3,
                ax=axes,
            )
            # backtest[i]['Volume'].plot(label=f'backtest Volume (forecast_horizon={forecast_horizon})', linewidth=1, ax=axes2[i])
            plt.legend()
