import pandas as pd
from sys import displayhook
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import src.Script_Model_Building as sc
import pickle
from pathlib import Path

data_prop = sc.open_csv_clean()  # Opens a csv file
pd.set_option('display.max_columns', 48)

# Splits apartments and houses data into training and testing
data_apartments, data_houses = sc.split_types(data_prop)
# Print the shape of apartments data
print('Shape of apartments data:', data_apartments.shape)
# Print the shape of houses data
print('Shape of houses data:',  data_houses.shape)

# Cleans apartments data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
data_apartments_clean = sc.apartment_clean(data_apartments)

# Cleans houses data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
data_houses_clean = sc.house_clean(data_houses)

# Splits apartments data frame into target array (price values) and features array (floor, amount of bedrooms, bathrooms, habirables quare, condition).
y_apart, X_apart = sc.targ_feature_apartment(data_apartments_clean)
# Splits houses data frame into target array (price values) and features array (amount of bedrooms, bathrooms, habirables quare, condition).
y_houses, X_houses = sc.targ_feature_house(data_houses_clean)

# Print the shape of target and features apartments and houses data
print('Shape of target apartments data:', y_apart.shape,
      'Shape of features apartments data:', X_apart.shape)
print('Shape of target houses data:', y_houses.shape,
      'Shape of features houses dara:', X_houses.shape)

# Visualizes relation between target ("Price") and features for apartments and houses. Saves plot into "output" folder.
sc.vis_data_apart(data_apartments_clean)
sc.vis_data_house(data_houses_clean)

# Spliting for apartment.
X_train_apart, X_test_apart, y_train_apart, y_test_apart = sc.split_data(
    X_apart, y_apart)
print('Shape of training and testing data for apartments:',
      X_train_apart.shape, X_test_apart.shape)  # print a shape of X_train_apart, X_test_apart.


# Spliting for houses.
X_train_houses, X_test_houses, y_train_houses, y_test_houses = sc.split_data(
    X_houses, y_houses)
print('Shape of training and testing data for houses:', X_train_houses.shape,
      X_test_houses.shape)  # print a shape of X_train_houses, X_test_houses.

# Applies XGBoos Regression model to the train and test data for apartments and houses.
y_pred_apart_xgbr, xgbr_apart, xgbr_tr_score_apart, xgbr_test_score_apart = sc.xgbr(
    X_train_apart, y_train_apart, X_test_apart, y_test_apart)
path_save_model_apart = Path.cwd() / "models" / "xgbr_apart.pickle"
with open(path_save_model_apart, 'wb') as file:
      pickle.dump(xgbr_apart, file)
y_pred_houses_xgbr, xgbr_houses, xgbr_tr_score_houses, xgbr_test_score_houses = sc.xgbr(
    X_train_houses, y_train_houses, X_test_houses, y_test_houses)
path_save_model_houses = Path.cwd() / "models" / "xgbr_houses.pickle"
with open(path_save_model_houses, 'wb') as file:
      pickle.dump(xgbr_houses, file)
# Compute mean squared error between target test data and target predicted data for apartments.
mse_apart_xgbr = mean_squared_error(y_test_apart, y_pred_apart_xgbr)
# Compute mean absolute error between target test data and target predicted data for apartments.
mae_apart_xgbr = mean_absolute_error(y_test_apart, y_pred_apart_xgbr)
# Compute mean squared error between target test data and target predicted data for houses.
mse_houses_xgbr = mean_squared_error(y_test_houses, y_pred_houses_xgbr)
# Compute mean absolute error between target test data and target predicted data for houses.
mae_houses_xgbr = mean_absolute_error(y_test_houses, y_pred_houses_xgbr)

# Print accuracy score on apartments and houses training and testing set (XGB Regressor) with normalized data, print mean squared error and mean absolute error.
print("Accuracy (Regressor score) on apartments training set (XGB Regressor) with normalized data:", xgbr_tr_score_apart)
print("Accuracy (Regressor score) on apartments testing set (XGB Regressor) with normalized data:",
      xgbr_test_score_apart)
print("Mean Squared Error (MSE) for apartments testing set (XGB Regressor) with normalized data:", mse_apart_xgbr)
print("Mean Absolute Error (MAE) for apartments testing set (XGB Regressor) with normalized data:", mae_apart_xgbr)

print("Accuracy (Regressor score) on houses training set (XGB Regressor) with normalized data:", xgbr_tr_score_houses)
print("Accuracy (Regressor score) on houses testing set (XGB Regressor) with normalized data:",
      xgbr_test_score_houses)
print("Mean Squared Error (MSE) for houses testing set (XGB Regressor) with normalized data:", mse_houses_xgbr)
print("Mean Absolute Error (MAE) for houses testing set (XGB Regressor) with normalized data:", mae_houses_xgbr)

# Makes a plot af relation between predited and actual values, saves it into "output" folder.
sc.scatter_pred(y_test_apart, y_pred_apart_xgbr, 'Apartments,',
                'XGB Regressor with normalized data')
sc.plt.savefig('output/plot_MB_9.png')
sc.plt.show()
sc.scatter_pred(y_test_houses, y_pred_houses_xgbr, 'Houses,',
                'XGB Regressor with normalized data')
sc.plt.savefig('output/plot_MB_10.png')
sc.plt.show()
