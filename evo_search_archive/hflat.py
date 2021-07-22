import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
# neural nets - decision trees are better
###################################################################


###################################################################
# HF-HAT config:
# config_example = {
#     'encoder': {
#         'encoder_embed_dim': 512,
#         'encoder_layer_num': 6,
#         'encoder_ffn_embed_dim': [3072, 3072, 3072, 3072, 3072, 3072],
#         'encoder_self_attention_heads': [8, 8, 8, 8, 8, 4],
#     }
# }
###################################################################

###################################################################
# Neural net to predict the latency given a set of features.
# Features are formed using the config parameters 
# of the Transformer. 
###################################################################
class Net(nn.Module):
    def __init__(self, feature_dim, hidden_dim, hidden_layer_num):
        super(Net, self).__init__()

        self.first_layer = nn.Linear(feature_dim, hidden_dim)

        self.layers = nn.ModuleList()

        for i in range(hidden_layer_num):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.predict = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.first_layer(x))

        for i in range(len(self.layers)):
            x = F.relu(self.layers[i](x))

        x = self.predict(x)

        return x


###################################################################
# Latency predictor class doing the following things:
# Initializes the Predictor
# Loads the Dataset
# Splits the Dataset
# Trains on Dataset
# Predicts latency given a config
###################################################################
class LatencyPredictor(object):
    def __init__(
            self, 
            feature_norm=[768, 12, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 70], 
            lat_norm=4, 
            ckpt_path = './latency_dataset/ckpts/nnlatcpkt.txt', 
            lat_dataset_path='./latency_dataset/sst2_gpu_gtx1080_final.csv', 
            feature_dim=27, 
            hidden_dim=200, 
            hidden_layer_num=3, 
            train_steps=5000, 
            bsz=128, 
            lr=1e-4 
        ):
        ###########################################
        # Leave Unchanged
        ###########################################
        self.dataset_path = lat_dataset_path
        self.feature_norm = np.array(feature_norm)
        self.lat_norm = lat_norm
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_layer_num = hidden_layer_num
        self.ckpt_path = ckpt_path

        self.dataset = None

        self.train_x = None
        self.train_y = None

        self.valid_x = None
        self.valid_y = None

        self.test_x = None
        self.test_y = None

        self.model = Net(self.feature_dim, self.hidden_dim, self.hidden_layer_num)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = torch.nn.MSELoss()

        self.train_steps = train_steps
        self.bsz = bsz

    def train(self):
        ###########################################
        # Leave Unchanged
        ###########################################
        for i in range(self.train_steps):
            sample_ind = random.sample(range(len(self.train_x)), k=self.bsz)
            sample_x = [self.train_x[sample_ind[k]] for k in range(self.bsz)]
            sample_y = [self.train_y[sample_ind[k]] for k in range(self.bsz)]

            sample_x_tensor = torch.Tensor(sample_x)
            sample_y_tensor = torch.Tensor(sample_y)

            prediction = self.model(sample_x_tensor).squeeze()

            loss = self.criterion(prediction, sample_y_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # validation
            if i % 100 == 0:
                with torch.no_grad():
                    sample_x_tensor = torch.Tensor(self.valid_x)
                    sample_y_tensor = torch.Tensor(self.valid_y)

                    prediction = self.model(sample_x_tensor).squeeze()
                    loss = self.criterion(prediction, sample_y_tensor)
                    print(f"Validation loss at {i} steps: {loss}")

        # test
        with torch.no_grad():
            sample_x_tensor = torch.Tensor(self.test_x)
            sample_y_tensor = torch.Tensor(self.test_y)
            prediction = self.model(sample_x_tensor).squeeze()
            loss = self.criterion(prediction, sample_y_tensor)
            # print(f"Predicted latency: {prediction}")
            # print(f"Real latency: {self.test_y}")
            print('Testing...')
            print(f"Loss: {loss}")

            # print(f"RMSE: {np.sqrt(self.criterion(self.lat_norm*sample_y_tensor, self.lat_norm*prediction))}")
            # print(f"MAPD: {torch.mean(torch.abs((sample_y_tensor - prediction) / sample_y_tensor))}")
            print("R2 score val : ",r2_score(sample_y_tensor*self.lat_norm, prediction*self.lat_norm))
            print("MSE score val: ", mean_squared_error(sample_y_tensor*self.lat_norm, prediction*self.lat_norm))

            #plot predicted vs actual:
            plt.scatter(x=sample_y_tensor*self.lat_norm, y=prediction*self.lat_norm, c='blue')
            plt.xlabel('actual (sec)')
            plt.ylabel('predicted (sec)')
            plt.savefig('nn_pred_vs_act.jpg')

        torch.save(self.model.state_dict(), self.ckpt_path)

    def load_ckpt(self):
        ###########################################
        # Leave Unchanged
        ###########################################
        self.model.load_state_dict(torch.load(self.ckpt_path))

    def predict_lat(self, config):
        ###########################################
        # Change only the get_config_features(config) function
        # Leave Unchanged
        ###########################################        
        with torch.no_grad():
            features = self.get_config_features(config)
            features_norm = np.array(features) / self.feature_norm

            prediction = self.model(torch.Tensor(features_norm)).item() * self.lat_norm

        return prediction

    def split(self):
        ###########################################
        # Leave Unchanged
        ###########################################
        sample_num = len(self.dataset['x'])
        train_num = int(np.floor(0.8 * sample_num))
        valid_num = int(np.floor(0.1 * sample_num))

        self.train_x = self.dataset['x'][:train_num]
        self.train_y = self.dataset['y'][:train_num]

        self.valid_x = self.dataset['x'][train_num:(train_num+valid_num)]
        self.valid_y = self.dataset['y'][train_num:(train_num+valid_num)]

        self.test_x = self.dataset['x'][(train_num+valid_num):]
        self.test_y = self.dataset['y'][(train_num+valid_num):]

    def read_dataset(self):
        ###########################################
        # Leave Unchanged
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
    
    def get_config_features(self, config):
        ###########################################
        # Removed the decoder features extracted 
        # by HAT authors
        # We need to change the this input representation in the future
        ###########################################
        features = []

        features.append(config['encoder']['encoder_embed_dim'])

        encoder_layer_num = config['encoder']['encoder_layer_num']
        features.append(encoder_layer_num)

        # encoder_ffn_embed_dim_mean = np.mean(config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num])
        # features.append(encoder_ffn_embed_dim_mean)
        features += config['encoder']['encoder_ffn_embed_dim'][:encoder_layer_num]+[-1]*(12-encoder_layer_num)
        features += config['encoder']['encoder_self_attention_heads'][:encoder_layer_num]+[-1]*(12-encoder_layer_num)
        features.append(60)

        # encoder_self_attention_heads_mean = np.mean(config['encoder']['encoder_self_attention_heads'][:encoder_layer_num])
        # features.append(encoder_self_attention_heads_mean)

        return features


###################################################################
# Main funciton:
# Parses all the arguments of the predictor
# trains predictor on dataset
# predicts the latency on some example config
###################################################################
if __name__=='__main__':
    # parser = configargparse.ArgumentParser()
    # # ?????CHECKOUT LATER??????
    # # parser.add_argument('--configs', required=True, is_config_file=True)
    # # parser.add_argument('--dataset-path')

    # parser.add_argument('--lat-dataset-path', type=str, default='./latency_dataset/encoder_latency_1.csv', help='the path to read latency dataset')
    # parser.add_argument('--feature-norm', type=float, nargs='+', default=[640, 6, 2048, 6], help='normalizing factor for each feature')
    # parser.add_argument('--lat-norm', type=float, default=200, help='normalizing factor for latency')
    # # Changed feature_dim to 4 as we use only encoder:
    # parser.add_argument('--feature-dim', type=int, default=4, help='dimension of feature vector')
    # # Changed hidden dim to 200:
    # parser.add_argument('--hidden-dim', type=int, default=200, help='hidden dimension of FC layers in latency predictor')
    # parser.add_argument('--hidden-layer-num', type=int, default=3, help='number of FC layers')
    # parser.add_argument('--ckpt-path', type=str, default='latency_dataset/ckpts/tmp.pt', help='path to save latency predictor weights')
    # # Changed the number of steps to 600:
    # parser.add_argument('--train-steps', type=int, default=600, help='latency predictor training steps')
    # parser.add_argument('--bsz', type=int, default=128, help='latency predictor training batch size')
    # parser.add_argument('--lr', type=float, default=1e-5, help='latency predictor training learning rate')
    # feature-norm =
    # args = parser.parse_args()
    # print(args)

    # predictor = LatencyPredictor(lat_dataset_path=args.lat_dataset_path,
    #                        feature_norm=args.feature_norm,
    #                        lat_norm=args.lat_norm,
    #                        feature_dim=args.feature_dim,
    #                        hidden_dim=args.hidden_dim,
    #                        hidden_layer_num=args.hidden_layer_num,
    #                        ckpt_path=args.ckpt_path,
    #                        train_steps=args.train_steps,
    #                        bsz=args.bsz,
    #                        lr=args.lr)

    predictor = LatencyPredictor()

    # # predictor.load_ckpt()
    predictor.read_dataset()
    predictor.split()
    predictor.train()
    print('Latency predictor training finished')

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

    predict = predictor.predict_lat(config_example)
    print(f'Example config: {config_example}')
    print(f'Example latency: {predict}')
