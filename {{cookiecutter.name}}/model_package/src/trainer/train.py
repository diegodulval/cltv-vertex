import fire


def train_model(gamma: float, data: str = "data.joblib", model: str = "model.joblib"):
    """Trains and saves a model given a data path and a model output path.
    Also takes a `gamma` argument for the svm model."""

    import os

    from joblib import dump, load
    from sklearn import svm
    import gcsfs

    # load the dataset
    with open(data, "rb") as f:
        digits = load(f)

    # flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))

    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=gamma, probability=True)

    # Learn the digits on the train subset
    clf.fit(data, digits.target)

    # Save the model as a pipeline artifact
    with open(model + ".joblib", "wb") as f:
        dump(clf, f)

    # Save model to AIP path to register it instantly
    if AIP_MODEL_DIR := os.getenv("AIP_MODEL_DIR"):
        model_path = os.path.join(AIP_MODEL_DIR, "model.joblib")
        fs = gcsfs.GCSFileSystem()
        with fs.open(model_path, "wb") as f:
            dump(clf, f)


if __name__ == "__main__":
    fire.Fire(train_model)
