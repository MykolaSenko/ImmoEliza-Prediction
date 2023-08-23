from pathlib import Path
import pickle
import numpy as np
import json


def open_reg(X_pred_pr):
    """
Depending on which type of property we are predicting the function opens a saved regressor from a pickle file in "models" directory
@param: X_pred_pr: input data(np.array)
@return reg_apart, re_houses: trained regressor
    """
    if X_pred_pr[0] == 'apartment':
        path_open_pickle_apart = Path.cwd() / "models" / "xgbr_apart.pickle"
        with open(path_open_pickle_apart, 'rb') as file:
            reg_apart = pickle.load(file)
        return reg_apart
    else:
        path_open_pickle_houses = Path.cwd() / "models" / "xgbr_houses.pickle"
        with open(path_open_pickle_houses, 'rb') as file:
            reg_houses = pickle.load(file)
        return reg_houses


def predict_new(X, reg):
    """
Makes a prediction accroding trained earlier data.
@param X: input data(np.array)
@param reg: XGboos regressor
@return: prediction data in json format
    """

    X = np.array(X)[1:]
    X = X.astype(np.float64)
    X = X.reshape(1, -1)
    y_pred_array = reg.predict(X)
    y_pred_dic = {
        "prediction": round(float(y_pred_array[0]), 2),
        "status_code": 200,
    }
    y_pred_json = json.dumps(y_pred_dic)
    return y_pred_json
