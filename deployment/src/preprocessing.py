import numpy as np


def processing(X):
    """
Preprocesses input data. The function checks type of data and deletes floor value if it is house, convert data into np.array
@param X: input data.
@return: preprocessed data(np.array)

    """
    X = X.dict()
    X['data']['property_type'] = X['data']['property_type'].lower()

    if X['data']['property_type'] == 'house':
        del X['data']['floor']
    else:
        pass

    X_values = list(X['data'].values())
    X_array = np.array(X_values)
    return X_array
