import fire

from .prepare_data import load_dataset
from .train import train_model


def workflow(data_dir="data.joblib", model_dir="model.joblib", gamma=0.1):
    """Workflow that executes all modelling pipeline steps.

    Args:
        data_dir (str, optional): Where loaded data will be saved. Defaults to "data.joblib".
        model_dir (str, optional): Model path where trained model will be saved. Defaults to "model.joblib".
        gamma (float, optional): Model parameter. Defaults to 0.1.
    """
    load_dataset(data_dir)
    train_model(gamma, data_dir, model_dir)


if __name__ == "__main__":
    fire.Fire(workflow)
