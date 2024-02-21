from canswim.model import CanswimModel
from loguru import logger
from dotenv import load_dotenv
import os
import time
from canswim.hfhub import HFHub


# main function
def main():

    canswim_model = CanswimModel()
    hfhub = HFHub()
    # Optionally download data if its not already available locally
    # NOTE: be careful not to override local data from previous searches!
    hfhub.download_data()
    load_dotenv(override=True)
    repo_id = os.getenv("repo_id", "ivelin/canswim")
    n_optuna_trials = int(os.getenv("n_optuna_trials", 100))

    def build_dummy_model():
        """Build a dummy model with max data load requirements"""
        canswim_model.build(
            input_chunk_length=252,
            output_chunk_length=42,
            hidden_size=2048,
            num_encoder_layers=3,
            num_decoder_layers=3,
            decoder_output_dim=8,
            temporal_decoder_hidden=64,
            use_layer_norm=True,
            use_reversible_instance_norm=True,
            dropout=0.2,
            optimizer_kwargs={"lr": 3.5e-05},
            save_checkpoints=False,  # checkpoint to retrieve the best performing model state,
            force_reset=False,
        )

    logger.info(f"Starting Model Search with {n_optuna_trials} trials...")
    # download market data from hf hub if it hasn't been downloaded already
    canswim_model.download_data(repo_id=repo_id)
    # build a max size dummy model to help data prep
    build_dummy_model()
    canswim_model.load_data()
    canswim_model.prepare_data()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    study = canswim_model.find_model(
        n_trials=n_optuna_trials, study_name=f"canswim-study-{timestr}"
    )
    # save model search results
    hfhub.upload_data()
    logger.info(f"Finished Model Search. Study: {study}")


if __name__ == "__main__":
    main()
