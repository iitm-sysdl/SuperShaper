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

class Predictor():
    def __init__(self, dataset_path=None, ckpt=None, pred_type='latency', model_type='xgb'):
        self.pred_type = pred_type
        self.ckpt = ckpt
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.keys = ["sample_hidden_size", "sample_num_attention_heads", "sample_intermediate_size", "sample_num_hidden_layers"]

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
            self.model = lgbm.Booster(model_file=self.ckpt_path)
        elif self.model_type == 'xgb':
            self.model = XGBRegressor()
            self.model.load_model(self.ckpt_path)
        else:
            raise NotImplementedError

    def store_ckpt(self, str_path):
        self.model.save_model(str_path)

    def read_dataset(self):
        assert self.dataset_path is not None

        df = pd.read_csv(self.dataset_path)

        ## Figure out how to normalize
        #df = df / df.max() ## Normalizing 
        
        df_features = df.drop([pred_type], axis=1)
        df_metric = df.drop(self.keys, axis=1)

        features = df_features.to_numpy()
        metric = df_metric.to_numpy()

        self.dataset = (features, metric)

        return self.dataset

    def train(self, dataset, split=0.7):
        dataset = dataset or self.dataset

        features, metric = dataset
        
        trainf = features[:int(split*len(features))]
        trainy = time[:int(split*len(features))]
        testf = features[int(split*len(features)):]
        testy = time[int(split*len(features)):]

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
        s_coefficient, pvalue = spearmanr(testy, testPredict)
        
        print("R2 Score: %f, Spearman Coefficient: %f, PValue: %f"%(r2_score_test, s_coefficient, pvalue))

        if self.model == 'xgb':
            print("Features of Importances")
            print(model.feature_importances_)
        
        self.plot(testy, test_predict)
    
    def plot(self, test_actual, test_predict):
        fig = go.Figure()

        testy = np.reshape(test_actual, (test_predict.shape[0],))
        fig.add_trace(go.Scatter(x=test_actual, y=test_predict, mode='markers'))

        fig.update_layout(xaxis_title="Actual Metric", yaxis_title="Predicted Metric", height=700, width=1000)
        
        fig.write_image("./metrics.pdf")
    
    def predict(self, feature_in):
        return self.model.predict(feature_in)


def test():
    pass


if __name__ == "__main__":
    test()
