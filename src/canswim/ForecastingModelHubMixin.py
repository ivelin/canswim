import os
from pathlib import Path
from typing import Dict, List, Optional, Type, TypeVar, Union
from huggingface_hub import ModelHubMixin, snapshot_download
from darts.models.forecasting.forecasting_model import ForecastingModel


class ForecastingModelHubMixin(ForecastingModel, ModelHubMixin):
    """
    Implementation of [`ModelHubMixin`] to provide model Hub upload/download capabilities to Darts Forecasting models.
    This mixin is based on the Hugging Face supported PyTorchModelHubMixin.
    See official Hugging Face Docs on Integration details:
    https://huggingface.co/docs/huggingface_hub/guides/integrations

    Example:

    ```python
    >>> from darts.models import TiDEModel
    >>> from darts... import ForecastingModelHubMixin

    >>> class MyModel(TiDEModel, ForecastingModelHubMixin):
    ...
    ...
    >>> model = MyModel()

    # Save model weights to local directory
    >>> model.save_pretrained("my-awesome-model")

    # Push model weights to the Hugging Face Hub
    >>> model.push_to_hub("my-awesome-model")

    # Download and initialize weights from the Hugging Face Hub
    >>> model = MyModel.from_pretrained("username/my-awesome-model")
    ```
    """

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        self.save(path=Path)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        revision: Optional[str],
        cache_dir: Optional[Union[str, Path]],
        force_download: bool,
        proxies: Optional[Dict],
        resume_download: bool,
        local_files_only: bool,
        token: Union[str, bool, None],
        map_location: str = "cpu",
        strict: bool = False,
        **model_kwargs,
    ):
        """Load and return pretrained model."""
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            local_dir = model_id
        else:
            local_dir = snapshot_download(
                repo_id=str(model_id),
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                token=token,
                local_files_only=local_files_only,
                local_dir=True,
            )
        model = ForecastingModel.load(path=local_dir)
        return model
