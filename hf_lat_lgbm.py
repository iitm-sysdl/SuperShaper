import random
import configargparse
import numpy as np
import lightgbm as lgb
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

###################################################################
# This latency predictor is taken and modified from the 
# HAT implementation of the same. 
#
# v1:
# As of now the training dataset for this latency_predictor is a 
# modified version of the data from the HAT implementation.
#
# We've restricted the dataset to the encoders only so that
# config matches our supertransformer design.
#
# v2: TODO
# Generate training data using our Supertransformer - write a 
# script for this
#
# Modify the prediction mechanism - suggestion use xgb instead of
# neural nets - decision trees are better - DONE
# This implementation uses lightGBM. The MSE is very good
###################################################################

class LatencyPredictor(object):
    #########################################
    # TODO: 
    # 1. add lgbm hyperparameters as cli
    #########################################
    def __init__(
            self, 
            # feature_norm=[640, 6, 2048, 6],
            feature_norm = [768, 12, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 70],
            #  lat_norm=200, 
            lat_norm = 4,
             ckpt_path = './latency_dataset/ckpts/lgb_1.txt', 
             lat_dataset_path='./latency_dataset/sst2_gpu_gtx1080_final.csv', 
             feature_dim=27
        ):
        self.dataset_path = lat_dataset_path
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.feature_dim = feature_dim
        self.ckpt_path = ckpt_path

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.model = None

    def load_ckpt(self):
        self.model = lgb.Booster(model_file=self.ckpt_path)
    
    def train(self):
        print('Training...')
        params = {
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
        # params = {
        #     'n_estimators': 500
        # }
        self.model = lgb.train(params= params,train_set= self.lgb_train_data, valid_sets=[self.lgb_test_data], num_boost_round=3000)
        print('Training of LightGBM finished...')

        # Test:
        print('Testing...')
        test_results = self.model.predict(self.test_x)
        print("R2 score val : ",r2_score(self.test_y*self.lat_norm ,test_results*self.lat_norm))
        print("MSE score val: ", mean_squared_error(self.test_y*self.lat_norm ,test_results*self.lat_norm))
        self.model.save_model(self.ckpt_path)


        for i in range(100):
            if self.test_y[i]*self.lat_norm < 2.0:
                print(f'predicted = {test_results[i]*self.lat_norm} and actual = {self.test_y[i]*self.lat_norm}')
        

        #plot predicted vs actual:
        plt.scatter(x=self.test_y*self.lat_norm, y=test_results*self.lat_norm, c='blue')
        plt.xlabel('actual(s)')
        plt.ylabel('predicted(s)')
        plt.savefig('lgb_pred_vs_act.jpg')

    def predict_lat(self, config):
        ###########################################
        # Predict using trained LightGBM
        # de-normalize and return
        ###########################################        
        features = [np.array(self.get_config_features(config)) / self.feature_norm]
        features_norm = np.array(features)

        prediction = self.model.predict(features)
        return prediction[0]*self.lat_norm

    def split(self):
        ###########################################
        # Convert train, valid, test sets to
        # lgb data
        ###########################################
        sample_num = len(self.dataset['x'])
        train_num = int(np.floor(0.9 * sample_num))
        valid_num = int(np.floor(0.0 * sample_num))
        test_num = int(np.floor(0.1 * sample_num))

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num+valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num+valid_num)]

        self.test_x = self.dataset['x'][(train_num+valid_num):(train_num+valid_num+test_num)]
        self.test_y = self.dataset['y'][(train_num+valid_num):(train_num+valid_num+test_num)]

        self.train_x = np.array(self.train_x)
        self.valid_x = np.array(self.train_x)
        self.test_x = np.array(self.test_x)
        self.train_y = np.array(self.train_y)
        self.valid_y = np.array(self.train_y)
        self.test_y = np.array(self.test_y)

        self.lgb_train_data = lgb.Dataset(self.train_x, label=self.train_y)
        self.lgb_valid_data = lgb.Dataset(self.valid_x, label=self.valid_y, reference=self.lgb_train_data)
        self.lgb_test_data = lgb.Dataset(self.test_x, label=self.test_y)

    def read_dataset(self):
        ###########################################
        # Left unchanged
        ###########################################
        features_norm_all = []
        lats_all = []
        with open(self.dataset_path, 'r') as fid:
            next(fid) # skip first line of CSV
            for line in fid:
                split_line = line.split(',')
                features = split_line[:self.feature_dim-1]+[split_line[-1]]
                # print(features)
                features_eval = list(map(eval, features))
                features_norm = np.array(features_eval) / self.feature_norm
                features_norm_all.append(features_norm)

                lats = [split_line[-2]]
                # print(lats)
                total_lat = eval(lats[0])
                lats_all.append(total_lat / self.lat_norm)
        tmp = list(zip(features_norm_all, lats_all))
        random.shuffle(tmp)
        features_norm_all, lats_all = zip(*tmp)
        self.dataset = {'x': features_norm_all, 'y': lats_all}
        # features_norm_all = []
        # lats_all = []
        # with open(self.dataset_path, 'r') as fid:
        #     next(fid) # skip first line of CSV
        #     for line in fid:
        #         features = line.split(',')[:self.feature_dim]
        #         features_eval = list(map(eval, features))
        #         features_norm = np.array(features_eval) / self.feature_norm
        #         features_norm_all.append(features_norm)

        #         lats = line.split(',')[self.feature_dim:]
        #         total_lat = eval(lats[0]) + eval(lats[1])
        #         lats_all.append(total_lat / self.lat_norm)
        # tmp = list(zip(features_norm_all, lats_all))
        # random.shuffle(tmp)
        # features_norm_all, lats_all = zip(*tmp)
        # self.dataset = {'x': features_norm_all, 'y': lats_all}
    
    def get_config_features(self, config):
        ###########################################
        # Removed the decoder features extracted 
        # by HAT authors
        # TODO: We need to change this input 
        # representation in the future.
        ###########################################
        features = []

        features.append(config['encoder']['encoder_embed_dim'])

        encoder_layer_num = config['encoder']['encoder_layer_num']
        features.append(encoder_layer_num)
        features += config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num]+[-1]*(12-encoder_layer_num)
        features += config['encoder']['encoder_self_attention_heads'][:encoder_layer_num]+[-1]*(12-encoder_layer_num)
        features.append(60)

        # encoder_ffn_embed_dim_mean = np.mean(config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num])
        # features.append(encoder_ffn_embed_dim_mean)

        # encoder_self_attention_heads_mean = np.mean(config['encoder']['encoder_self_attention_heads'][:encoder_layer_num])
        # features.append(encoder_self_attention_heads_mean)

        return features


if __name__=='__main__':
    predictor = LatencyPredictor(ckpt_path='./latency_dataset/ckpts/lgb_cpu_1.txt', lat_dataset_path='./latency_dataset/sst2_cpu.csv')
    # predictor.load_ckpt()
    predictor.read_dataset()
    predictor.split()
    predictor.train()
    print('Latency predictor training finished...')

    predictor.load_ckpt()
    config_example = {
        'encoder': {
            'encoder_embed_dim': 768,
            'encoder_layer_num': 12,
            'encoder_ffn_embed_dim': [3072]*12,
            'encoder_self_attention_heads': [12]*12,
        }
    }
    predict = predictor.predict_lat(config_example)
    print(f'Example config: {config_example}')
    print(f'Example latency: {predict}')
    config_example = {
        'encoder': {
            'encoder_embed_dim': 360,
            'encoder_layer_num': 2,
            'encoder_ffn_embed_dim': [512]*2,
            'encoder_self_attention_heads': [6]*2,
        }
    }
    # print(predictor.get_config_features(config_example))
    predict = predictor.predict_lat(config_example)
    print(f'Example config: {config_example}')
    print(f'Example latency: {predict}')