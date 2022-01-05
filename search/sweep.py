import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime

import xgboost as xgb
from xgboost.callback import EarlyStopping
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# import train test split
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


import wandb
from wandb.xgboost import wandb_callback

# Login
# wandb.login()

data = pd.read_csv("data.csv")
# split data into 70% train and 30% test
train, test = train_test_split(data, test_size=0.3)
# take everything except last columns for X_train and rest for y_train
X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
# take everything except last columns for X_test and rest for y_test
X_valid, y_valid = test.iloc[:, :-1], test.iloc[:, -1]


def train():
    with wandb.init() as run:
        bst_params = {
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "booster": run.config.booster,
            "learning_rate": run.config.learning_rate,
            "gamma": run.config.gamma,
            "max_depth": run.config.max_depth,
            "min_child_weight": run.config.min_child_weight,
            "eval_metric": ["rmse", "mae"],
            "tree_method": "gpu_hist",
        }

        # Initialize the XGBoostClassifier
        xgbmodel = xgb.XGBRegressor(**bst_params)

        # Train the model, using the wandb_callback for logging
        xgbmodel.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            callbacks=[
                wandb_callback(
                )
            ],
            verbose=False,
        )

        preds = xgbmodel.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(y_valid, preds))
        print("RMSE: %f" % (rmse))
        wandb.log({"Valid_RMSE": rmse})
        print("R2: %f" % (r2_score(y_valid, preds)))
        wandb.log({"Valid_R2": r2_score(y_valid, preds)})
        print("Spearman: %f" % (spearmanr(y_valid, preds)[0]))
        wandb.log({"Valid_Spearman": spearmanr(y_valid, preds)[0]})


sweep_config = {
    "name": "hyperparam_search",
    "method": "random",
    "parameters": {
        "booster": {"values": ["gbtree", "gblinear"]},
        "learning_rate": {"min": 0.001, "max": 1.0},
        "gamma": {"min": 0.001, "max": 1.0},
        "max_depth": {"values": [3, 5, 7]},
        "min_child_weight": {"min": 1, "max": 150},
        "early_stopping_rounds": {
            "values": [
                10,
                20,
                40,
                40,
            ]
        },
    },
}

sweep_id = wandb.sweep(sweep_config, project="hyperparam_search")

wandb.agent(sweep_id, project="hyperparam_search", function=train)

