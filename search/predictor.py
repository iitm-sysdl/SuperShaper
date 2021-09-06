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
sentences = ["hello how are you", "i am fine"]
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
inputs = tokenizer.encode_plus(sentences, return_tensors="pt")
plotly.io.orca.config.save()

# prepring model
bert_config = AutoConfig.from_pretrained("bert-base-uncased")



## Move this to utils later ## 
def convert_to_dict(string):
    _dict = json.loads(
        string.replace("BertConfig ", "").replace("\n", "").replace('""', '"')
    )
    return BertConfig(**_dict)


def generate_dataset(features_file, pred_type='perplexity'):
    df = pd.read_csv(features_file) 
    df["config"] = df["config"].map(convert_to_dict)

    feature_dict = {
        "sample_hidden_size": [],
        "sample_num_attention_heads": [],
        "sample_intermediate_size": [],
        "sample_num_hidden_layers": [],
        "params": [],
        #"macs": [],
        pred_type: [],

    }
    
    for index, row in df.iterrows():
        cfg = row["config"]
        params = calculate_params_from_config(cfg)
        #print("Adding row %d to dataset"%(index))

        for key in feature_dict.keys():
            if key == pred_type:
                feature_dict[key].append(row[pred_type])
            elif key == 'params':
                feature_dict[key].append(params)
            elif key == 'macs':
                model = custom_bert.BertForMaskedLM(config=cfg)
                model.set_sample_config(cfg)
                macs = profile_macs(model, inputs["input_ids"])

                feature_dict[key].append(macs)
            else:
                attr = getattr(cfg, key)
                feature_dict[key].append(attr)
    
    return pd.DataFrame.from_dict(feature_dict)

def row_mapper(row):
    return np.hstack([row["sample_hidden_size"], row["sample_num_attention_heads"], row["sample_intermediate_size"], row["sample_num_hidden_layers"], row["params"]])


class Predictor():
    def __init__(self, dataset_path=None, ckpt=None, pred_type='latency', model_type='xgb'):
        self.pred_type = pred_type
        self.ckpt = ckpt
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.keys = ["sample_hidden_size", "sample_num_attention_heads", "sample_intermediate_size", "sample_num_hidden_layers", "params"]

        if model_type == 'lgbm': 
            self.model = lgbm.Booster() 
        elif model_type == 'xgb':
            self.model = XGBRegressor()

        if self.model_type == 'lgbm':
            self.lgb_params = {
                'boosting_type': 'gbdt',
                'objective': 'regression',
                'metric': {'l2'},
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': 0,
                # 'n_estimators': 1000,
            }

    def load_ckpt(self):
        assert self.ckpt is not None

        if self.model_type == 'lgbm':
            self.model = lgbm.Booster(model_file=self.ckpt)
        elif self.model_type == 'xgb':
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
        #df = df / df.max() ## Normalizing 
        
        df_features = df.drop([self.pred_type], axis=1)
        df_metric = df.drop(self.keys, axis=1)
        
        #df_metric = df_metric / df_metric.max()

        metric = df_metric.to_numpy()
        
        df_features["merged_features"] = df_features.apply(row_mapper, axis=1)

        features = np.vstack(df_features["merged_features"].tolist())
        
        self.dataset = (features, metric)

        return self.dataset

    def train(self, dataset=None, split=0.7, plot=False):
        dataset = dataset or self.dataset

        features, metric = dataset
        
        trainf = features[:int(split*len(features))]
        trainy = metric[:int(split*len(features))]
        testf = features[int(split*len(features)):]
        testy = metric[int(split*len(features)):]

        if self.model_type == 'lgbm':
            lgbm_train_data = self.model.Dataset(trainf, label=trainy)
            lgbm_test_data = self.model.Dataset(testf, label=test, reference=lgbm_train_data)
 
            self.model.train(params = self.lgb_params, train_set=lgbm_train_data, valid_sets=[lgbm_test_data], num_boost_round=3000)

        elif self.model_type == 'xgb':
            self.model.fit(trainf, trainy)  

        ## Testing the trained model ## 
        test_predict = self.model.predict(testf)
        testScore = math.sqrt(mean_squared_error(testy, test_predict))

        r2_score_test = sklearn.metrics.r2_score(testy, test_predict)
        s_coefficient, pvalue = spearmanr(testy, test_predict)
        
        print(f"Number of Training Examples: {trainf.shape} , Test Data: {testf.shape}")
        print("R2 Score: %f, Spearman Coefficient: %f, PValue: %f"%(r2_score_test, s_coefficient, pvalue))

        if self.model_type == 'xgb':
            print("Features of Importances")
            print(self.model.feature_importances_)
       	
        if plot: 
            self.plot(testy, test_predict)
    
    def plot(self, test_actual, test_predict):
        fig = go.Figure()

        testy = np.reshape(test_actual, (test_predict.shape[0],))

        fig.add_trace(go.Scatter(x=testy, y=test_predict, mode='markers'))

        fig.update_layout(xaxis_title="Actual Metric", yaxis_title="Predicted Metric", height=700, width=1000)
        
        fig.write_image("./metrics_"+self.pred_type+"_.pdf")
    
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
        default='xgb',
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


    args = parser.parse_args()
    
    return args


 

def learn_predictor(args):
    df = generate_dataset(args.input_file_name_or_path, pred_type=args.prediction_type)
    predictor = Predictor(pred_type=args.prediction_type, model_type=args.model_type)
    predictor.read_dataset(df)
    predictor.train(plot=args.plot)
    predictor.store_ckpt(args.output_file_name_or_path+'.'+args.model_type)

if __name__ == "__main__":
    args = parse_args()
    learn_predictor(args)
