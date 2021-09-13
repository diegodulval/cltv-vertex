import os

import fire
from kfp.components import InputPath, OutputPath
from kfp.v2 import compiler
from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component
from kfp.v2.google.client import AIPlatformClient


def load_dataset_vtx_component(config_path, base_image, aliz_aip_project):

    @component(base_image=base_image)
    def load_dataset_vtx(config_path: str,
                         aliz_aip_project: str,
                         cleaned_data: Output[Dataset],
                         eda_dir: Output[Metrics]):
        """Function that loads a predefined dataset and saves it in the defined path in joblib format."""

        import os
        import pandas as pd
        import mlflow

        # from aliz.aip.ml.mlflow import setup_mlflow_env
        # setup_mlflow_env(aliz_aip_project)

        # --------- CLEAN --------------
        import numpy as np
        import pandas as pd
        from scipy.stats import norm
        import sys
        from pprint import pprint


        def data_clean(data: pd.DataFrame, config: dict):
            # Drop columns which are configured to drop
            cols = [col[3:] if col[:3] == "f__" else col for col in data.columns]
            print(cols)
            print(f"Number of columns: {len(data.columns)}")
            print(f"Number of rows: {len(data)}")

            # TODO: remove when new data with the correct date is obtained.
            print(f"Changing `{config['date']}` column to randomly generated dates...")
            def random_dates(start, end, n=10):
                start_u = start.value//10**9
                end_u = end.value//10**9
                return pd.to_datetime(np.random.randint(start_u, end_u, n), unit='s')

            start = pd.to_datetime('2016-01-01')
            end = pd.to_datetime('2021-12-12')
            dates = random_dates(start, end, n=len(data))
            data[config['date']] = dates

            data.columns = cols
            for col in config['drop']:
                try:
                    del data[col]
                except:
                    pass


        def fake_data(data: pd.DataFrame, config: dict):
            """Copy one date to another and add some random noise."""
            fake = data[data[config['date']] == pd.to_datetime('2017-08-01')].copy()
            fake[config['date']] = pd.to_datetime('2017-07-31')
            for col in config['attributes']:
                if col in config['drop'] or config['attributes'][col]['type'] != 'numeric':
                    continue

                jitter = data[col].astype(float).std() * 0.5  # 50% of Standard deviation as jitter
                fake[col] += np.random.normal(size=fake.shape[0]) * jitter

            fake = pd.concat([fake, data], axis=0).reset_index(drop=True)
            fake[config['date']] = pd.to_datetime(fake[config['date']])

            return fake

        # --------- DATA --------------

        import os
        import sys
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from sklearn.base import TransformerMixin, BaseEstimator
        from tqdm import tqdm
        import google.auth
        from google.cloud import bigquery
        from google.cloud import bigquery_storage


        def _query_df(query_string, project_id='mlops-featurestore-sandbox'):
            return pd.read_gbq(query_string, project_id=project_id, use_bqstorage_api=True)


        def query_df(query_string, project_id='mlops-featurestore-sandbox'):
            # Explicitly create a credentials object. This allows you to use the same
            # credentials for both the BigQuery and BigQuery Storage clients, avoiding
            # unnecessary API calls to fetch duplicate authentication tokens.
            credentials, project_id_ = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

            #assert project_id_ == project_id, f"Project id should be '{project_id}'"

            # Make clients.
            bqclient = bigquery.Client(credentials=credentials, project=project_id,)
            bqstorageclient = bigquery_storage.BigQueryReadClient(credentials=credentials) # try to add project_id here also

            dataframe = (
                bqclient.query(query_string)
                .result()
                .to_dataframe(bqstorage_client=bqstorageclient)
            )

            return dataframe


        def data_load_bq(query: str = None):
            import pandas_gbq
            try:
                print("pd.__version__", pd.__version__)
                print("pandas_gbq.__version__", pandas_gbq.__version__)
                print("bigquery.__version__", bigquery.__version__)
                print("bigquery_storage.__version__", bigquery_storage.__version__)
            except Exception as e:
                print("Exception", e)

            q_columns = '*'
            query = query or f'SELECT {q_columns} FROM `mlops-featurestore-sandbox.ga_features_dev.aip_features_WIDE` WHERE ABS(MOD(FARM_FINGERPRINT(entity_id),100)) < 1'
            return query_df(query)


        def data_load(data_path: str, config: dict):
            df = pd.read_parquet(data_path)

            if config:
                missing = set(df.columns) - set(config['attributes'])
                if missing:
                    print("Columns missing from configuration:", missing)
                missing = set(config['attributes']) - set(df.columns)
                if missing:
                    print("Columns missing from data:", missing)

            return df


        def convert_to_int(df, col):
            replace(df, col, '', 0)
            replace(df, col, 'null', 0)
            replace(df, col, None, 0)
            astype(df, col, np.float64)  # We need float, for NaN-s.


        def binarize(df, col, tsh, compare_higher=False):
            df[col] = (df[col] > tsh).astype(int) if compare_higher else (df[col] < tsh).astype(int)


        def convert_to_str(df, col):
            replace(df, col, None, '')
            apply(df, col, str)


        def apply(df, col, fn):
            df[col] = df[col].apply(fn)


        def replace(df, col, old, new):
            if old is None:
                df[col] = df[col].fillna(new)
            else:
                df.loc[df[col] == old, col] = new


        def astype(df, col, type):
            df[col] = df[col].astype(type)


        class AipPreprocessTransformer(TransformerMixin, BaseEstimator):
            """Wrapper for `data_preprocess` as sklearn Transformer."""

            def __init__(self, config):
                self.config = config

            def fit(self, X):
                return self

            def transform(self, X: pd.DataFrame):
                """We expect X to be DataFrame, because the config file is based on column names."""
                data_preprocess(X, self.config, use_tqdm=False)

                return X


        def data_preprocess(df: pd.DataFrame, config: dict, use_tqdm: bool = False):
            for col in tqdm(config['attributes'], disable=(not use_tqdm)):
                print(f"Preprocess data '{col}'")
                if col in config['drop']:
                    continue
                if 'preprocess' in config['attributes'][col]:
                    if 'copy_from' in config['attributes'][col]:
                        df[col] = df[config['attributes'][col]['copy_from']].copy()
                    for kwargs in config['attributes'][col]['preprocess']:

                        fn = L[kwargs['_fn']]

                        kwargs = {k: kwargs[k] for k in kwargs if not k.startswith('_')}
                        try:
                            fn(df, col, **kwargs)
                        except ValueError as e:
                            print(col, e)
                        except TypeError as e:
                            print(col, e)
                        except KeyError as e:
                            print(col, e)


        def data_assessment(df: pd.DataFrame, config: dict, output_dir: str, use_tqdm: bool = False):
            def assess_categorical(col: str):
                data: pd.Series = df[col]

                r = {
                    'Record count': data.size,
                    'Missing': data.isna().sum(),
                    'Unique values': data.unique().size,
                }

                for i, (name, value) in enumerate(data.value_counts().iloc[:5].items()):
                    r[f"%d. most common value '%s' count" % (i+1, str(name))] = value

                return r

            def assess_numeric(col: str):
                data = df[col]

                std = data.std()
                q75 = data.quantile(0.75)
                q25 = data.quantile(0.25)
                mean = data.mean()

                r = {
                    'Record count': data.size,
                    'Missing': data.isna().sum(),
                    'Unique values': data.unique().size,
                    'Min': data.min(),
                    'Q01': data.quantile(0.01),
                    'Q05': data.quantile(0.05),
                    'Q25': q25,
                    'Mean': mean,
                    'Median': data.quantile(0.50),
                    'Q75': q75,
                    'Q95': data.quantile(0.95),
                    'Q99': data.quantile(0.99),
                    'Max': data.max(),
                    'Std': std,
                    'IQR': (q75 - q25),
                    'CV': (std / mean if mean > 0 else None)
                }

                return r

            def save_results(result: dict, col: str):
                filename = f'{output_dir}/{col}.csv'
                results = pd.DataFrame.from_dict(result, orient='index')
                results.columns, results.index.name = ['Value'], 'Result'
                results.to_csv(filename, index=True, header=True)

            Path(output_dir).mkdir(exist_ok=True, parents=True)
            for col in tqdm(config['attributes'], disable=(not use_tqdm)):
                print(f"Data assessment '{col}'")
                try:
                    if col in config['drop']:
                        continue
                    if 'categorical' == config['attributes'][col]['type']:
                        save_results(assess_categorical(col), col)
                    elif 'numeric' == config['attributes'][col]['type']:
                        save_results(assess_numeric(col), col)
                except:
                    print(f"'{col}' is problematic.")
                    pass


        # ---------- UTIL -------------

        import os
        import yaml
        import gcsfs


        def config_template(data, save_filename=None):
            """Generate config template as a dict or yaml string."""

            def _is_numeric(s):
                if str(s.dtype).startswith('int') or str(s.dtype).startswith('float'):
                    return True

                if s.apply(lambda x: 'Decimal' in str(type(x))).any():
                    return True

                return False

            config = dict()
            config['target'] = '- TODO -'
            config['date'] = '- TODO -'
            config['context'] = '- TODO -'
            config['drop'] = []

            config['attributes'] = {}
            for col in data.columns:
                col = str(col)
                dtype = str(data[col].dtype)
                print(dtype, ' - ', col)

                config['attributes'][col] = {}

                numeric = _is_numeric(data[col])
                config['attributes'][col]['type'] = 'numeric' if numeric else 'categorical'

                if _is_numeric(data[col]):
                    config['attributes'][col]['type'] = 'numeric'
                    config['attributes'][col]['preprocess'] = [
                        {'_fn': 'convert_to_int'},
                    ]
                else:
                    config['attributes'][col]['type'] = 'categorical'

                for integer_attribute in ['__Cnt_', '_Click__Sum_', '_View__Sum_', 'SocialInteractions__Sum_',
                                        'Transaction_Shipping__Sum_']:
                    if integer_attribute in col:
                        config['attributes'][col]['preprocess'] = [
                            {'_fn': 'convert_to_int'},
                        ]

            if save_filename is None:
                return config
            else:
                assert not os.path.exists(save_filename), f"Target filename '{save_filename}' should not exist."
                with open(save_filename, 'w') as f:
                    f.write(str(yaml.dump(config)))
                    f.close()


        def load_config(config_path):
            # Open file handle GCS or local
            f = gcsfs.GCSFileSystem().open(config_path, "r") if config_path[:5] == 'gs://' else open(config_path)
            # Parse yaml
            config = yaml.safe_load(f)
            # Close handle
            f.close()

            return config


        from sklearn.pipeline import Pipeline
        from pathlib import Path
        G = globals()
        L = locals()
        print("G")
        print(G)
        print("L")
        print(L)

        with mlflow.start_run() as mlrun:
            # Load Config
            config = load_config(config_path)

            # Load data
            print("Load data", flush=True)
            data = data_load_bq()

            # Clean data
            print("Clean data", flush=True)
            data_clean(data, config)

            # Data Preprocessing
            print("Preprocess data", flush=True)
            pipeline = Pipeline(steps=[('aip_preprocess', AipPreprocessTransformer(config))])
            data = pipeline.fit_transform(data)

            # Data Assessment
            print("Data assessment", flush=True)
            Path(eda_dir.path).mkdir(parents=True, exist_ok=True)
            data_assessment(data, config, eda_dir.path, use_tqdm=False)
            mlflow.log_artifacts(eda_dir.path)

            print("Save data", flush=True)
            data.to_parquet(cleaned_data.path)
            print("Saved file size (bytes):", os.path.getsize(cleaned_data.path), flush=True)

            mlflow.log_artifact(cleaned_data.path)

    return load_dataset_vtx(config_path=config_path, aliz_aip_project=aliz_aip_project)


if __name__ == '__main__':
    fire.Fire(load_dataset)
