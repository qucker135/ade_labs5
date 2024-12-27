import os
import numpy as np
import pandas as pd
from scipy.stats import moment
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

PATH = os.path.join('datasets', 'Folds5x2_pp.xlsx')

if __name__ == "__main__":
    df = pd.read_excel(PATH, sheet_name='Sheet1')
    print(df.head())
    print("-----------------")
    print(df.info())
    print("-----------------")
    print(df.describe())
    print("-----------------")
    # compute variances of each column
    print("Wariances:")
    print(f"{df.var()}")
    print("-----------------")
    # compute third moment of each column
    print("Third (central) moments:")
    print(f"{df.apply(moment, moment=3)}")
    print("-----------------")
    # compute skewness of each column
    print("Skewness:")
    print(f"{df.skew()}")
    print("-----------------")
    # compute fourth moment of each column
    print("Fourth (central) moments:")
    print(f"{df.apply(moment, moment=4)}")
    print("-----------------")
    # compute kurtosis (excess) of each column
    print("Kurtosis (excess):")
    print(f"{df.kurt()}")
    print("-----------------")
    # compute coefficient of variation of each column
    print("Coefficient of variation:")
    print(f"{df.std() / df.mean() * 100.0}")
    print("-----------------")
    
    # draw histograms (save to file)
    df.hist(bins=50, figsize=(20,15))
    plt.savefig('histograms.png')

    # compute Pearson correlation matrix
    print("Pearson correlation matrix:")
    print(df.corr())
    print("-----------------")

    # compute Spearman correlation matrix
    print("Spearman correlation matrix:")
    print(df.corr(method='spearman'))
    print("-----------------")

    # compute covariance matrix
    print("Covariance matrix:")
    print(df.cov())
    print("-----------------")

    # Linear regression (whole dataset)
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=['AT', 'V', 'AP', 'RH'])
    reg = LinearRegression().fit(X, y)
    print("Linear regression coefficients (whole dataset):")
    print(f"Intercept: {reg.intercept_}")
    print(f"Coefficients: {reg.coef_}")
    # compute R^2
    print(f"R^2: {reg.score(X, y)}")
    print(f"MSE: {((y - reg.predict(X)) ** 2).mean()}")
    print(f"RMSE: {((y - reg.predict(X)) ** 2).mean() ** 0.5}")
    print(f"MAE: {abs(y - reg.predict(X)).mean()}")
    print("-----------------")
    
    # Quadratic regression (whole dataset)
    X['AT^2'] = X['AT'] ** 2
    X['V^2'] = X['V'] ** 2
    X['AP^2'] = X['AP'] ** 2
    X['RH^2'] = X['RH'] ** 2
    X['AT*V'] = X['AT'] * X['V']
    X['AT*AP'] = X['AT'] * X['AP']
    X['AT*RH'] = X['AT'] * X['RH']
    X['V*AP'] = X['V'] * X['AP']
    X['V*RH'] = X['V'] * X['RH']
    X['AP*RH'] = X['AP'] * X['RH']
    reg = LinearRegression().fit(X, y)
    print("Quadratic regression coefficients (whole dataset):")
    print(f"Intercept: {reg.intercept_}")
    print(f"Coefficients: {reg.coef_}")
    # compute R^2
    print(f"R^2: {reg.score(X, y)}")
    print(f"MSE: {((y - reg.predict(X)) ** 2).mean()}")
    print(f"RMSE: {((y - reg.predict(X)) ** 2).mean() ** 0.5}")
    print(f"MAE: {abs(y - reg.predict(X)).mean()}")
    print("-----------------")

    # train-test split
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=64)

    # Standardize data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Linear regression (train-test split)
    reg = LinearRegression().fit(X_train, y_train)
    print("Linear regression coefficients (train-test split):")
    print(f"Intercept: {reg.intercept_}")
    print(f"Coefficients: {reg.coef_}")
    # compute R^2
    print(f"R^2: {reg.score(X_test, y_test)}")
    print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
    print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
    print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
    print("-----------------")

    # Quadratic regression (train-test split)
    X_train['AT^2'] = X_train['AT'] ** 2
    X_train['V^2'] = X_train['V'] ** 2
    X_train['AP^2'] = X_train['AP'] ** 2
    X_train['RH^2'] = X_train['RH'] ** 2
    X_train['AT*V'] = X_train['AT'] * X_train['V']
    X_train['AT*AP'] = X_train['AT'] * X_train['AP']
    X_train['AT*RH'] = X_train['AT'] * X_train['RH']
    X_train['V*AP'] = X_train['V'] * X_train['AP']
    X_train['V*RH'] = X_train['V'] * X_train['RH']
    X_train['AP*RH'] = X_train['AP'] * X_train['RH']

    X_test['AT^2'] = X_test['AT'] ** 2
    X_test['V^2'] = X_test['V'] ** 2
    X_test['AP^2'] = X_test['AP'] ** 2
    X_test['RH^2'] = X_test['RH'] ** 2
    X_test['AT*V'] = X_test['AT'] * X_test['V']
    X_test['AT*AP'] = X_test['AT'] * X_test['AP']
    X_test['AT*RH'] = X_test['AT'] * X_test['RH']
    X_test['V*AP'] = X_test['V'] * X_test['AP']
    X_test['V*RH'] = X_test['V'] * X_test['RH']
    X_test['AP*RH'] = X_test['AP'] * X_test['RH']

    reg = LinearRegression().fit(X_train, y_train)
    print("Quadratic regression coefficients (train-test split):")
    print(f"Intercept: {reg.intercept_}")
    print(f"Coefficients: {reg.coef_}")
    # compute R^2
    print(f"R^2: {reg.score(X_test, y_test)}")
    print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
    print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
    print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
    print("-----------------")

    # add white noise to PE
    NOISE_LVL = 0.1
    df['PE'] += NOISE_LVL * df['PE'].std() * np.random.randn(df.shape[0])
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=64)

    # Standardize data
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    # Linear regression (train-test split, noisy PE)
    reg = LinearRegression().fit(X_train, y_train)
    print("Linear regression coefficients (train-test split, noisy PE):")
    print(f"Intercept: {reg.intercept_}")
    print(f"Coefficients: {reg.coef_}")
    # compute R^2
    print(f"R^2: {reg.score(X_test, y_test)}")
    print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
    print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
    print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
    print("-----------------")

    regulatisations = [0.1, 1.0, 10.0, 100.0]

    for alpha in regulatisations:
        # Ridge regression (train-test split, noisy PE)
        reg = Ridge(alpha=alpha).fit(X_train, y_train)
        print(f"Ridge regression coefficients (train-test split, noisy PE, alpha={alpha}):")
        print(f"Intercept: {reg.intercept_}")
        print(f"Coefficients: {reg.coef_}")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        print("-----------------")

        # Lasso regression (train-test split, noisy PE)
        reg = Lasso(alpha=alpha).fit(X_train, y_train)
        print(f"Lasso regression coefficients (train-test split, noisy PE, alpha={alpha}):")
        print(f"Intercept: {reg.intercept_}")
        print(f"Coefficients: {reg.coef_}")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        print("-----------------")
    
