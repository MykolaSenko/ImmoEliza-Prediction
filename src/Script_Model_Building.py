import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn import preprocessing
import xgboost as xg



def open_csv_clean():
    '''
    Opens csv file.
    @param path_to_data_csv: path to csv file where data for properties from the second stage of the project are stored
    @return data_prop (pd.DataFrame): Pandas data frame of property objects
    '''
    path_to_data_csv = Path.cwd() / "data" / "_properties_data_clean.csv"
    data_prop = pd.read_csv(path_to_data_csv)
    return data_prop


def split_types(data_prop):
    '''
    Splits data frame with properties into two data frames: apartments and houses.
    @return data_apartments (pd.DataFrame): Pandas data frame of apartments.
    @return data_houses(pd.DataFrame): Pandas data frame of houses.
    '''
    data_apartments = data_prop[data_prop['type'] == 'Apartment']
    data_houses = data_prop[data_prop['type'] == 'House']
    return data_apartments, data_houses


def apartment_clean(data_apartments):
    '''
    Cleans apartments data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
    @return data_apartments_clean (pd.DataFrame): Cleaned up Pandas data frame of apartments.
    '''

    data_apartments_clean = data_apartments[[
        'price', 'floor', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']]

    rows_with_no_information = data_apartments_clean.loc[(data_apartments_clean['floor'] == 'No_information') |
                                                         (data_apartments_clean['bedroomCount'] == 'No_information') |
                                                         (data_apartments_clean['bathroomCount'] == 'No_information') |
                                                         (data_apartments_clean['netHabitableSurface'] == 'No_information') |
                                                         (data_apartments_clean['condition'] == 'No_information')]

    rows_to_drop = data_apartments_clean.index.isin(
        rows_with_no_information.index)
    data_apartments_clean = data_apartments_clean[~rows_to_drop]

    data_apartments_clean['condition'].replace(
        {'To_be_done_up': 0, 'To_restore': 1, 'To_renovate': 2, 'Just_renovated': 3, 'Good': 4, "As_new": 5}, inplace=True)

    data_apartments_clean = data_apartments_clean.apply(
        pd.to_numeric, errors='coerce')
    data_apartments_clean[['price', 'floor', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']] = data_apartments_clean[[
        'price', 'floor', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']].astype('int64')

    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['floor'] < 17)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['bedroomCount'] < 7)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['bathroomCount'] <= 3)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['netHabitableSurface'] < 350)]
    data_apartments_clean = data_apartments_clean[(
        data_apartments_clean['price'] < 2500000)]

    return data_apartments_clean


def house_clean(data_houses):
    '''
    Cleans houses data: deelete rows with 'No_information values, replaces non numerical values into numerical, converts all columns into intergers, delete outlayers.
    @return data_houses_clean (pd.DataFrame): Cleaned up Pandas data frame of houses.
    '''
    data_houses_clean = data_houses[[
        'price', 'bedroomCount', 'bathroomCount', 'netHabitableSurface', 'condition']]

    rows_with_no_information_1 = data_houses_clean.loc[(data_houses_clean['bedroomCount'] == 'No_information') |
                                                       (data_houses_clean['bathroomCount'] == 'No_information') |
                                                       (data_houses_clean['netHabitableSurface'] == 'No_information') |
                                                       (data_houses_clean['condition'] == 'No_information')]

    rows_to_drop = data_houses_clean.index.isin(
        rows_with_no_information_1.index)
    data_houses_clean = data_houses_clean[~rows_to_drop]

    data_houses_clean['condition'].replace(
        {'To_be_done_up': 0, 'To_restore': 1, 'To_renovate': 2, 'Just_renovated': 3, 'Good': 4, "As_new": 5}, inplace=True)

    data_houses_clean = data_houses_clean.apply(pd.to_numeric, errors='coerce')
    data_houses_clean[['price', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']] = data_houses_clean[[
        'price', 'bedroomCount', 'netHabitableSurface', 'bathroomCount', 'condition']].astype('int64')

    data_houses_clean = data_houses_clean[(
        data_houses_clean['price'] < 3500000)]
    data_houses_clean = data_houses_clean[(
        data_houses_clean['bathroomCount'] <= 8)]
    data_houses_clean = data_houses_clean[(
        data_houses_clean['bedroomCount'] <= 12)]
    data_houses_clean = data_houses_clean[(
        data_houses_clean['netHabitableSurface'] < 700)]

    return data_houses_clean


def targ_feature_apartment(df_apart):
    '''
    Splits apartments data frame into target array (price values) and features array (floor, amount of bedrooms, bathrooms, habirables quare, condition).
    @param df_apart (pd.DataFraame): data frame of apartments.
    @return y_ap (nparray): array of terget values.
    @return X_ap (nparray): array of featured values.
    '''
    y_ap = df_apart['price'].values
    X_ap = df_apart[['floor', 'bedroomCount',
                     'netHabitableSurface', 'bathroomCount', 'condition']].values
    return y_ap, X_ap


def targ_feature_house(df_houses):
    '''
    Splits houses data frame into target array (price values) and features array (amount of bedrooms, bathrooms, habirables quare, condition).
    @param df_houses (pd.DataFraame): data frame of houses.
    @return y_h (nparray): array of terget values.
    @return X_h (nparray): array of featured values.
    '''
    y_h = df_houses['price'].values
    X_h = df_houses[['bedroomCount', 'bathroomCount',
                     'netHabitableSurface', 'condition']].values
    return y_h, X_h


def vis_data_apart(df_apart):
    '''
    Visualizes relation between target ("Price") and features for apartments. Saves plot into "output" folder.
    @param df_apart (pd.DataFraame): data frame of apartments.
    '''
    # Doubles the size of the figure
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    fig.suptitle(
        'Relations between target ("Price") and features for apartments', fontsize=16)

    # Sort the data by x-values
    sorted_data = df_apart.sort_values('floor')

    axes[0, 0].scatter(sorted_data['floor'], sorted_data['price'])
    axes[0, 0].set_xlabel('Floor')
    axes[0, 0].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[0, 0].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_apart.sort_values('bedroomCount')

    axes[0, 1].scatter(sorted_data['bedroomCount'], sorted_data['price'])
    axes[0, 1].set_xlabel('Amount of bedrooms')
    axes[0, 1].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[0, 1].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_apart.sort_values('netHabitableSurface')

    axes[1, 0].scatter(sorted_data['netHabitableSurface'],
                       sorted_data['price'])
    axes[1, 0].set_xlabel('Net Habitable Surface')
    axes[1, 0].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[1, 0].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_apart.sort_values('bathroomCount')

    axes[1, 1].scatter(sorted_data['bathroomCount'], sorted_data['price'])
    axes[1, 1].set_xlabel('Amount of bathrooms')
    axes[1, 1].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[1, 1].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_apart.sort_values('condition')

    axes[2, 0].scatter(sorted_data['condition'], sorted_data['price'])
    axes[2, 0].set_xlabel('Condition of porperties')
    axes[2, 0].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[2, 0].tick_params(axis='x', rotation=90)

    plt.savefig('output/plot_MB_1.png')

    plt.show()

def vis_data_house(df_houses):
    '''
    Visualizes relation between target ("Price") and features for houses. Saves plot into "output" folder.
    @param df_houses (pd.DataFraame): data frame of houses.
    '''
    # Doubles the size of the figure
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

    fig.suptitle(
        'Relations between target ("Price") and features for houses', fontsize=16)

    # Sort the data by x-values
    sorted_data = df_houses.sort_values('bedroomCount')

    axes[0, 0].scatter(sorted_data['bedroomCount'], sorted_data['price'])
    axes[0, 0].set_xlabel('Amount of bedrooms')
    axes[0, 0].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[0, 0].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_houses.sort_values('netHabitableSurface')

    axes[0, 1].scatter(sorted_data['netHabitableSurface'],
                       sorted_data['price'])
    axes[0, 1].set_xlabel('Net Habitable Surface')
    axes[0, 1].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[0, 1].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_houses.sort_values('bathroomCount')

    axes[1, 0].scatter(sorted_data['bathroomCount'], sorted_data['price'])
    axes[1, 0].set_xlabel('Amount of bathrooms')
    axes[1, 0].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[1, 0].tick_params(axis='x', rotation=90)

    # Sort the data by x-values
    sorted_data = df_houses.sort_values('condition')

    axes[1, 1].scatter(sorted_data['condition'], sorted_data['price'])
    axes[1, 1].set_xlabel('Condition of porperties')
    axes[1, 1].set_ylabel('Price')
    # Rotate x-axis label vertically
    axes[1, 1].tick_params(axis='x', rotation=90)

    plt.savefig('output/plot_MB_2.png')

    plt.show()


def split_data(X, y):
    '''
    Splits apartments and houses data into training and testing
    @param X(nparray): features data of apartments and houses.
    @param y(nparray): target data of apartments and houses.
    @return X_train, X_test, y_train, y_test(nparray): features and target data for apartments and houses.
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def xgbr(X_train, y_train, X_test, y_test):
    '''
    Applies XGBoos Regression model to the train and test data for apartments and houses.
    @param X_train, y_train, X_test, y_test (nparray): normalized features train and test data for apartment and houses target data for apartment and houses.
    @return y_pred (nparray): predicted target data for apartments and houses.
    @return xgb_r.score(X_train, y_train), xgb_r.score(X_test, y_test): accuracy score for train and test data for apartments and houses.
    '''
    xgb_r = xg.XGBRegressor(objective='reg:squarederror',
                            n_estimators=10)
    xgb_r.fit(X_train, y_train)
    y_pred = xgb_r.predict(X_test)
    return y_pred, xgb_r, xgb_r.score(X_train, y_train), xgb_r.score(X_test, y_test)


def scatter_pred(y_test, y_pred, type, model):
    '''
    Makes a plot af relation between predited and actual values.
    '''
    plt.scatter(y_test, y_pred)
    plt.title(type + ' ' + model)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
