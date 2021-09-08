import fire


def load_dataset(data: str = "data.joblib"):
    """Function that loads a predefined dataset and saves it in the defined path in joblib format."""

    from sklearn import datasets
    from joblib import dump

    digits = datasets.load_digits()

    with open(data, 'wb') as f:
        dump(digits, f)


if __name__ == '__main__':
    fire.Fire(load_dataset)
