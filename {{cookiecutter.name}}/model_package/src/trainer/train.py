import os
from typing import NamedTuple

import fire
from kfp.components import InputPath, OutputPath
from kfp.dsl.types import GCSPath
from kfp.v2.dsl import Dataset, Input, Metrics, Model, Output, component


def train_model_vtx_component(config_path, cleaned_data, base_image, aliz_aip_project):

    @component(base_image=base_image)
    def train_model_vtx(config_path: str,
                        cleaned_data: Input[Dataset],
                        aliz_aip_project: str,
                        model_path: Output[Model]) -> NamedTuple("model_uri", [("model_uri", str)]):
        import mlflow
        import pandas as pd
        from collections import namedtuple
        import os
        import pickle
        import time
        import warnings
        from math import sqrt
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from catboost import CatBoostRegressor, CatBoostClassifier
        from scipy.stats import pearsonr
        from sklearn.metrics import matthews_corrcoef as mcc, \
            balanced_accuracy_score, \
            confusion_matrix, \
            matthews_corrcoef, \
            log_loss, f1_score, precision_score, recall_score, \
            accuracy_score
        from sklearn.metrics import \
            mean_squared_error, mean_absolute_error, \
            median_absolute_error, \
            explained_variance_score, max_error, \
            r2_score
        from pathlib import Path
        import shutil

        # from aliz.aip.ml.mlflow import setup_mlflow_env
        # setup_mlflow_env(aliz_aip_project)


        warnings.filterwarnings('ignore')
        sns.set(rc={'figure.figsize': (12, 9)})
        sns.set(style="darkgrid")


        def get_features(config, categorical_only=False):
            features = set(config['attributes'].keys()) - {config['context'], config['date'], config['target']} \
                - set(config['drop'])

            if categorical_only:
                return [f for f in features if config['attributes'][f]['type'] == 'categorical']
            else:
                return features


        def cv_split(data, config):
            #features = get_features(config)


            features = list(set(data.columns) - {config['context'], config['date'], config['target']} - set(config['drop']))
            print(features)
            print(len(features))
            print(config['target'])
            print("--------")

            """
            mask = data[config['date']] >= pd.to_datetime(config['cross_validation']['split_date'])
            X_train, y_train = data.loc[~mask, features], data.loc[~mask, config['target']]
            X_test, y_test = data.loc[mask, features], data.loc[mask, config['target']]
            """
            from sklearn.model_selection import train_test_split
            X, y = data.loc[:, features], data.loc[:, config['target']]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
            print(len(X_train))
            print(len(X_test))

            return X_train, X_test, y_train, y_test


        def tnr(y_true, y_pred):
            """True negative rate."""
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return tn / (tn + fp)


        def train_eval(estimator_type,
                       X_train, y_train,
                       X_test, y_test,
                       params,
                       target_label,
                       output_dir):
            """Train a CatBoostRegressor model
            Args:
                estimator_type (str): Type of model, 'classifier' or 'regressor'.
                X_train (np.array): training features
                y_train (np.array): training target value
                X_test (np.array): tets features
                y_test (np.array): test target value
                params (object): hyperparameters of a CatBoostRegressor model
                target_label (str): target label
                output_dir (str): plots and CSVs output dir
                binary_bounds (tuple): boundaries to use for binary annotation
                class1_below_tsh (bool): Class 1 examples are below or above threshold, for ThresholdBinarizer
            Returns: model trained, training & test performances, threshold binarizer summary
            """
            model_params = params.copy()  # make a copy
            is_clf = estimator_type == 'classifier'
            model = CatBoostClassifier(**model_params) if is_clf else CatBoostRegressor(**model_params)

            print("Train data:")
            #print(pd.concat([X_train, y_train], axis=1).describe().T)



            # Training
            print('Training...')
            t0 = time.time()
            model.fit(X_train, y_train)
            print('   Training time: %.2f s' % (time.time() - t0))

            # Evaluate training performance
            print('Evaluating training performance...')
            t0 = time.time()
            y_pred = model.predict(X_train)
            y_pred = pd.Series(y_pred, index=X_train.index)

            training_perf = eval_clf(y_train, y_pred, target_label) if is_clf else eval_reg(y_train, y_pred, target_label)
            plot(y_train.values, y_pred, target_str='%s training set' % target_label, output_dir=output_dir)
            print('   Evaluation time: %.2f s' % (time.time() - t0))

            # Evaluate test performance
            print('Evaluating test performance...')
            t0 = time.time()
            y_pred = model.predict(X_test)
            y_pred = pd.Series(y_pred, index=X_test.index)
            test_perf = eval_clf(y_test, y_pred, target_label) if is_clf else eval_reg(y_test, y_pred, target_label)
            plot(y_test.values, y_pred, target_str='%s test set' % target_label, output_dir=output_dir)
            print('   Evaluation time: %.2f s' % (time.time() - t0))

            return model, training_perf, test_perf


        # low-level metrics
        def _RMSE(y_true, y_pred):
            return sqrt(mean_squared_error(y_true, y_pred))


        def _r2(y_true, y_pred):
            return pearsonr(y_true, y_pred)[0] ** 2


        def eval_reg(y_true, y_pred):
            """Compute model metrics
            https://en.wikipedia.org/wiki/Coefficient_of_determination
            """
            evaluation = {'RMSE': _RMSE,
                          'MAE': mean_absolute_error,
                          'MedAE': median_absolute_error,
                          'explained_variance': explained_variance_score,
                          'max_error': max_error,
                          'R2': r2_score,
                          'r2': _r2}
            metrics = {}
            for k, v in evaluation.items():
                metrics[k] = v(y_true, y_pred)

            return pd.DataFrame.from_dict(metrics, orient='index', columns=['metric value'])


        def plot(y_true, y_pred, target_str, output_dir):
            """Create and save plots."""
            N = len(y_true)
            df_GT = pd.DataFrame([y_true, ['groundtruth'] * N], index=[target_str, 'nature']).T
            df_pred = pd.DataFrame([y_pred.to_numpy(), ['prediction'] * N], index=[target_str, 'nature']).T
            df = pd.concat([df_GT, df_pred], axis=0)

            for label_, df_ in df.groupby('nature'):
                sns.distplot(df_[target_str], hist=False, rug=False, label=label_)

            plt.title('{} - groundtruth vs. prediction'.format(target_str))

            plot_filename = os.path.join(output_dir, target_str + '.png')
            plt.savefig(plot_filename)
            plt.close()


        def plot_timeseries(y_true, y_pred, y_index, target_str, output_dir):
            # Plot time series
            df = pd.DataFrame(data={
                'groundtruth': np.asarray(y_true),
                'prediction': np.asarray(y_pred),
                't_stamp': pd.to_datetime(y_index),
            })
            df['prediction error'] = df['prediction'] - df['groundtruth']
            df['normalized abs. error'] = ((df['prediction'] - df['groundtruth']) / df['groundtruth']).apply(np.abs)

            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 9))

            plot_df = df.set_index('t_stamp')[['groundtruth', 'prediction']]
            sns.lineplot(data=plot_df, ax=ax1)
            ax1.set(title=f"{target_str} prediction vs. ground truth", xlabel=None)
            # Hide tick labels
            # https://stackoverflow.com/questions/20936658/how-to-hide-ticks-label-in-python-but-keep-the-ticks-in-place/26428792
            labels = [item.get_text() for item in ax1.get_xticklabels()]
            empty_string_labels = [''] * len(labels)
            ax1.set_xticklabels(empty_string_labels)

            plot_df = df.set_index('t_stamp')[['prediction error']]
            sns.lineplot(data=plot_df, ax=ax2)
            ax2.set(title=f"{target_str} prediction error", xlabel=None)
            # Hide tick labels
            labels = [item.get_text() for item in ax2.get_xticklabels()]
            empty_string_labels = [''] * len(labels)
            ax2.set_xticklabels(empty_string_labels)

            plot_df = df.set_index('t_stamp')[['normalized abs. error']]
            sns.lineplot(data=plot_df, ax=ax3)
            ax3.set(title=f"{target_str} normalized absolute error", xlabel=None)

            plot_filename = os.path.join(output_dir, target_str + '.png')
            plt.savefig(plot_filename)
            plt.close()


        def modelling_results(eval_model, inference_model, training_perf, test_perf, binarizer_summary,
                              feature_list, setting_list, target_label, output_dir):
            """Export modelling artifacts: model pickle file, metrics, feature importances. Also does simple sanity check
            whether the most important features are settings or not: Out of one third of the most important features, at least
            half should be settings. If this criteria is not met, a warning message is displayed.

            Args:
                eval_model (CatBoostRegressor): Trained model on subset of data
                inference_model(CatBoostRegressor): Trained model on full data
                training_perf (pandas.DataFrame): Metrics on training dataset
                test_perf (pandas.DataFrame): Metrics on test dataset
                binarizer_summary (pandas.Series): Threshold binarization metrics.
                feature_list (list): List of all features remaining after cleaning
                setting_list (list): List of setting features remaining after cleaning
                target_label (str): Name of the target label
                output_dir (str): Path to save modelling artifacts

            Returns:
                  None
            """
            # Save model
            model_filename = os.path.join(output_dir, f"{target_label} eval model.pkl")
            pickle.dump(eval_model, open(model_filename, "wb"))
            model_filename = os.path.join(output_dir, f"{target_label} inference model.pkl")
            pickle.dump(inference_model, open(model_filename, "wb"))

            # Save name of features
            feature_list_filename = os.path.join(output_dir, f"{target_label} feature list.csv")
            pd.Series(feature_list).to_csv(feature_list_filename, header=False, index=False)

            # Save features importances
            feature_importances = pd.DataFrame(
                [eval_model.feature_importances_,
                 np.arange(len(feature_list)),
                 [(f in setting_list) for f in feature_list]],
                index=['importance', 'index_position', 'isSetting'],
                columns=feature_list
            ).T
            feature_importances.sort_values(by='importance', ascending=False, inplace=True)
            feature_importances.to_csv(os.path.join(output_dir, f"{target_label} feature importances.csv"),
                                       index_label='feature')

            # Simple sanity check
            top_features_count = round(feature_importances.shape[0] * 0.333)
            if feature_importances[:top_features_count]['isSetting'].astype(int).mean() < 0.5:
                print(f"Warning: {target_label} model's most important features are not settings!")

            # Save eval metrics
            training_perf.to_csv(os.path.join(output_dir, f"{target_label} training metrics.csv"), index_label='metric')
            test_perf.to_csv(os.path.join(output_dir, f"{target_label} test metrics.csv"), index_label='metric')

            # Save ThresholdBinarizer result
            if binarizer_summary is not None:
                binarizer_summary.to_csv(os.path.join(output_dir, f"{target_label} threshold binarizer.csv"), header=False)

        def eval_clf(y_true, y_pred, sample_weight=None):
            evaluation = {
                'MCC': matthews_corrcoef,
                'logLoss': log_loss,
                'f1': f1_score,
                'precision': precision_score,
                'recall': recall_score,
                'accuracy': accuracy_score,
            }
            metrics = {}
            for k, v in evaluation.items():
                metrics[k] = v(y_true, y_pred)
            return pd.DataFrame.from_dict(metrics, orient='index', columns=['metric value'])


        def do_modelling(data, config, output_dir):
            # Train-test split


            # workaround

            all_features = set(data.columns) - {config['context'], config['date'], config['target']}
            num_features = data._get_numeric_data().columns
            cat_features = list(set(all_features) - set(num_features))

            data[num_features].fillna(0, inplace = True)
            data[cat_features].fillna("", inplace = True)
            print("---data----")
            print(data.sum)

            print("catNA")
            data[cat_features].isna().sum().sum()

            print("numtNA")
            data[num_features].isna().sum().sum()

            print("targetNA")
            data[config['target']].isna().sum().sum()


            print(data)
            print(data.info())


            X_train, X_test, y_train, y_test = cv_split(data, config)


            # Train and eval model
            params = {
                'verbose': 50,
                'allow_writing_files': False,
                'random_seed': 55,
                'random_strength': 999,
                'cat_features': cat_features,
                **config['modelling']['kw_params'],
            }

            shutil.rmtree(output_dir, ignore_errors=True)
            Path(output_dir).mkdir(parents=True, exist_ok=False)
            estimator_type = config['modelling']['estimator_type']
            model, train_perf, test_perf = train_eval(estimator_type, X_train, y_train, X_test, y_test, params, config['target'], output_dir)
            print("Train performance")
            print(train_perf)
            print("Test performance")
            print(test_perf)

            return model, train_perf, test_perf






        # ---------------
        import yaml
        import gcsfs

        def load_config(config_path):
            # Open file handle GCS or local
            f = gcsfs.GCSFileSystem().open(config_path, "r") if config_path[:5] == 'gs://' else open(config_path)
            # Parse yaml
            config = yaml.safe_load(f)
            # Close handle
            f.close()

            return config


        with mlflow.start_run() as mlrun:
            # Load config and dataset
            config = load_config(config_path)
            data = pd.read_parquet(cleaned_data.path)


            # Train model and evaluate performance
            model, train_perf, test_perf = do_modelling(data, config, model_path.path)
            artifact_path = "model"
            #mlflow.catboost.log_model(model, artifact_path, registered_model_name='CHURN')  # todo: cookiecutter beállítja valahogy
            # this above line is problematic
            mlflow.log_artifacts(model_path.path)

            uri = namedtuple("model_uri", ["model_uri"])
            return uri(mlrun.info.artifact_uri + "/" + artifact_path)

    return train_model_vtx(config_path=config_path, cleaned_data=cleaned_data, aliz_aip_project=aliz_aip_project)


if __name__ == "__main__":
    fire.Fire(train_model)
