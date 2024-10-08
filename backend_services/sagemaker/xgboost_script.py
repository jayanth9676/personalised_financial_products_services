import argparse
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def model_fn(model_dir):
    model = xgb.Booster()
    model.load_model(os.path.join(model_dir, "xgboost-model"))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])
    parser.add_argument('--num-round', type=int, default=100)
    args, _ = parser.parse_known_args()

    train_df = pd.read_csv(os.path.join(args.train, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.validation, "validation.csv"))

    y_train = train_df['loan_approved']
    X_train = train_df.drop('loan_approved', axis=1)
    y_val = val_df['loan_approved']
    X_val = val_df.drop('loan_approved', axis=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dval = xgb.DMatrix(X_val_scaled, label=y_val)

    params = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "objective": "binary:logistic",
        "eval_metric": "auc"
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=[(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds=10
    )

    model.save_model(os.path.join(args.model_dir, "xgboost-model"))