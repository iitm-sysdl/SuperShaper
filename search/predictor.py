import os, sys

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(dir_path, os.pardir))
sys.path.insert(0, parent_dir_path)

import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBRegressor
import sklearn
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import math
from sklearn.utils import shuffle
import plotly.graph_objects as go
import json
from transformers import BertConfig
from utils import calculate_params_from_config
from torchprofile import profile_macs
from transformers import AutoConfig, AutoTokenizer
from custom_layers import custom_bert, custom_mobile_bert
import argparse
import plotly

# sentences = ["hello how are you", "i am fine"]
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# inputs = tokenizer.encode_plus(sentences, return_tensors="pt")
plotly.io.orca.config.save()

# prepring model
# bert_config = AutoConfig.from_pretrained("bert-base-uncased")


## Move this to utils later ##
def convert_to_dict(string):
    _dict = json.loads(
        string.replace("BertConfig ", "").replace("\n", "").replace('""', '"')
    )
    return BertConfig(**_dict)


def print_feature_importance(features2shape, feature_importances):
    idx = 0
    for feature, rows in features2shape.items():
        importances = feature_importances[idx : idx + rows]
        importances = [round(i * 100, 2) for i in importances]
        print(f"{feature} : {importances}")
        idx += rows


def generate_dataset(
    features_file, pred_type="perplexity", layerdrop=False, use_params=False
):
    df = pd.read_csv(features_file)
    df["config"] = df["config"].map(convert_to_dict)

    feature_dict = {
        "sample_hidden_size": [],
        "sample_num_attention_heads": [],
        "sample_intermediate_size": [],
        "sample_num_hidden_layers": [],
        # "params": [],
        # "macs": [],
        pred_type: [],
    }

    if layerdrop:
        feature_dict["depth_features"] = []

    if use_params:
        feature_dict["params"] = []

    for index, row in df.iterrows():
        cfg = row["config"]
        params = calculate_params_from_config(cfg)
        # print("Adding row %d to dataset"%(index))

        for key in feature_dict.keys():
            if key == pred_type:
                feature_dict[key].append(row[pred_type])
            elif key == "params":
                feature_dict[key].append(params)
            elif key == "macs":
                model = custom_bert.BertForMaskedLM(config=cfg)
                model.set_sample_config(cfg)
                macs = profile_macs(model, inputs["input_ids"])

                feature_dict[key].append(macs)
            else:

                # Not using number of layers as a feature as its xgb is overfitting
                # to this.

                # if key == "sample_num_hidden_layers" and layerdrop:
                #     # during layerdrop, the number of layer reduces
                #     num_layers = getattr(cfg, "sample_num_hidden_layers") - sum(
                #         getattr(cfg, "depth_features")
                #     )
                #     feature_dict[key].append(num_layers)
                #     continue

                # if key == "sample_hidden_size" and layerdrop:
                #     depth_features = getattr(cfg, "depth_features")
                #     hidden_sizes = getattr(cfg, "sample_hidden_size")
                #     changed_hidden_sizes = []
                #     for idx, hidden_size in enumerate(hidden_sizes):
                #         if depth_features[idx] == 0:
                #             changed_hidden_sizes.append(hidden_size)
                #         else:
                #             changed_hidden_sizes.append(0)
                #     feature_dict[key].append(changed_hidden_sizes)
                #     continue
                attr = getattr(cfg, key)
                feature_dict[key].append(attr)

    return pd.DataFrame.from_dict(feature_dict)


def row_mapper(row):
    list_to_stack = [
        row["sample_hidden_size"],
        row["sample_num_attention_heads"],
        row["sample_intermediate_size"],
    ]

    if "depth_features" in row.keys():
        list_to_stack.append(row["depth_features"][: row["sample_num_hidden_layers"]])

    if "params" in row.keys():
        list_to_stack.append(row["params"])

    if "macs" in row.keys():
        list_to_stack.append(row["macs"])

    list_to_stack.append(row["sample_num_hidden_layers"])

    return np.hstack(list_to_stack)


class Predictor:
    def __init__(
        self,
        args_dict,
        dataset_path=None,
        ckpt=None,
        pred_type="latency",
        model_type="xgb",
        layerdrop=False,
        use_params=False,
    ):
        self.pred_type = pred_type
        self.ckpt = ckpt
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.layerdrop = layerdrop
        self.use_params = use_params
        self.features2shape = {}

        self.keys = [
            "sample_hidden_size",
            "sample_num_attention_heads",
            "sample_intermediate_size",
        ]
        if self.layerdrop:
            self.keys.append("depth_features")
        if self.use_params:
            self.keys.append("params")

        self.keys.append("sample_num_hidden_layers")

        if model_type == "lgbm":
            self.model = lgb.Booster()
        elif model_type == "xgb":
            if args_dict == {}:
                self.model = XGBRegressor()
            else:
                self.model = XGBRegressor(
                    max_depth=args_dict["max_depth"],  # {5, 9, 10, 14}
                    n_estimators=args_dict["n_estimators"],
                    min_child_weight=args_dict["min_child_weight"],  # {1, 5, 6, 10}
                    subsample=args_dict["subsample"],  # {1, 0.8, 0.6, 0.3}
                    alpha=args_dict["alpha"],  # [.3, .2, .1, .05, .01, .005]
                    eta=args_dict["eta"],  # [.3, .2, .1, .05, .01, .005]
                    seed=args_dict["seed"],
                )

        if self.model_type == "lgbm":
            self.lgb_params = {
                "boosting_type": "gbdt",
                "objective": "regression",
                "metric": {"l2"},
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "verbose": 0,
                # 'n_estimators': 1000,
            }

    def load_ckpt(self):
        assert self.ckpt is not None

        if self.model_type == "lgbm":
            self.model = lgb.Booster(model_file=self.ckpt)
        elif self.model_type == "xgb":
            self.model = XGBRegressor()
            self.model.load_model(self.ckpt)
        else:
            raise NotImplementedError

    def store_ckpt(self, str_path):
        self.model.save_model(str_path)

    def read_dataset(self, df=None):

        if self.dataset_path is not None:
            assert df is None
            df = pd.read_csv(self.dataset_path)

        ## Figure out how to normalize
        # df = df / df.max() ## Normalizing

        df_features = df.drop([self.pred_type], axis=1)
        df_metric = df.drop(self.keys, axis=1)

        for key in df_features.columns:
            if isinstance(df_features.iloc[0][key], list):
                self.features2shape[key] = len(df_features.iloc[0][key])
            else:
                self.features2shape[key] = 1
        # df_metric = df_metric / df_metric.max()

        metric = df_metric.to_numpy()

        df_features["merged_features"] = df_features.apply(row_mapper, axis=1)

        features = np.vstack(df_features["merged_features"].tolist())

        self.dataset = (features, metric)

        return self.dataset

    def train(self, dataset=None, split=0.7, plot=False):
        print("Training ...")
        dataset = dataset or self.dataset

        features, metric = dataset

        trainf = features[: int(split * len(features))]
        trainy = metric[: int(split * len(features))]
        testf = features[int(split * len(features)) :]
        testy = metric[int(split * len(features)) :]

        if self.model_type == "lgbm":
            lgbm_train_data = self.model.Dataset(trainf, label=trainy)
            lgbm_test_data = self.model.Dataset(
                testf, label=testy, reference=lgbm_train_data
            )

            self.model.train(
                params=self.lgb_params,
                train_set=lgbm_train_data,
                valid_sets=[lgbm_test_data],
                num_boost_round=3000,
            )

        elif self.model_type == "xgb":
            self.model.fit(trainf, trainy)

        ## Testing the trained model ##
        test_predict = self.model.predict(testf)
        testScore = math.sqrt(mean_squared_error(testy, test_predict))

        r2_score_test = sklearn.metrics.r2_score(testy, test_predict)
        s_coefficient, pvalue = spearmanr(testy, test_predict)

        print(f"Number of Training Examples: {trainf.shape} , Test Data: {testf.shape}")
        print(
            "R2 Score: %f, Spearman Coefficient: %f, PValue: %f"
            % (r2_score_test, s_coefficient, pvalue)
        )

        if self.model_type == "xgb":
            print("Features of Importances")
            # print(self.model.feature_importances_)

            print_feature_importance(
                self.features2shape, self.model.feature_importances_
            )

        if plot:
            self.plot(testy, test_predict)

    def plot(self, test_actual, test_predict):
        fig = go.Figure()

        testy = np.reshape(test_actual, (test_predict.shape[0],))

        fig.add_trace(go.Scatter(x=testy, y=test_predict, mode="markers"))

        import pandas as pd
        df_testy = pd.DataFrame(testy)
        df_testP = pd.DataFrame(test_predict)

        df_testy.to_csv("ryzen_test_actual.csv")
        df_testP.to_csv("ryzen_test_predict.csv")

        fig.update_layout(
            xaxis_title="Actual Metric",
            yaxis_title="Predicted Metric",
            height=700,
            width=1000,
        )

        fig.write_image("./metrics_" + self.pred_type + "_.pdf")

    def predict(self, feature_in):
        return self.model.predict(feature_in)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learn a predictor for latency and/or perplexity"
    )

    parser.add_argument(
        "--input_file_name_or_path",
        type=str,
        required=True,
        help="The file name of the output",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        required=True,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="xgb",
        help="The type of cost model used",
    )
    parser.add_argument(
        "--output_file_name_or_path",
        type=str,
        required=True,
        help="Path to store the learnt model",
    )

    parser.add_argument(
        "--plot",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--layer_drop",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--use_params",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--min_child_weight",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--subsample",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--eta",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    return args


def learn_predictor(args):
    df = generate_dataset(
        args.input_file_name_or_path,
        pred_type=args.prediction_type,
        layerdrop=args.layer_drop,
        use_params=args.use_params,
    )
    print(df.iloc[0])

    args_dict = {
        "max_depth": args.max_depth,
        "n_estimators": args.n_estimators,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "alpha": args.alpha,
        "eta": args.eta,
        "seed": args.seed,
    }

    predictor = Predictor(
        args_dict=args_dict,
        pred_type=args.prediction_type,
        model_type=args.model_type,
        layerdrop=args.layer_drop,
        use_params=args.use_params,
    )
    predictor.read_dataset(df)
    predictor.train(plot=args.plot)
    predictor.store_ckpt(args.output_file_name_or_path + "." + args.model_type)


if __name__ == "__main__":
    args = parse_args()
    learn_predictor(args)
