import pandas as pd
import numpy as np

import pickle

# /home/xgboosting/mysite/


def make_prediction(date, model):
    date = date + ' 12:00:00'

    df = pd.read_csv('Bishkekdata.csv')

    pred_df = df[df['Date'] == date].drop(['AQI', 'Date'], axis=1)

    boolean_cols = pred_df.select_dtypes(include=['bool']).columns.tolist()
    # Convert boolean columns to numeric
    pred_df[boolean_cols] = pred_df[boolean_cols].astype(int)

    if model == 'xgboost':
        reg = pickle.load(open('xgboost.pkl', 'rb'))

        y_pred = reg.predict(pred_df)
        y_pred = np.clip(y_pred, 0, 5)

        return round(y_pred[0])
    elif model == 'random':
        reg = pickle.load(open('random.pkl', 'rb'))

        y_pred = reg.predict(pred_df)
        y_pred = np.clip(y_pred, 0, 5)

        return y_pred
    elif model == 'linear':
        reg = pickle.load(open('linear.pkl', 'rb'))

        y_pred = reg.predict(pred_df)
        y_pred = np.clip(y_pred, 0, 5)

        return y_pred

    elif model == 'neural':
        reg = pickle.load(open('/home/xgboosting/mysite/neural.pkl', 'rb'))

        y_pred = reg.predict(pred_df)
        y_pred = np.clip(y_pred, 0, 5)

        return y_pred
