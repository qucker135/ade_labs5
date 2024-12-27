def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt

PATH = os.path.join('datasets', 'Folds5x2_pp.xlsx')

if __name__ == "__main__":
    df = pd.read_excel(PATH, sheet_name='Sheet1')
    # data preparation
    X = df[['AT', 'V', 'AP', 'RH']]
    y = df['PE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=64)
    # standardize
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Linear regression
    reg = LinearRegression().fit(X_train, y_train)
    print("Linear regression coefficients:")
    print(f"Intercept: {reg.intercept_}")
    print(f"Coefficients: {reg.coef_}")
    # compute R^2
    print(f"R^2: {reg.score(X_test, y_test)}")
    print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
    print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
    print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
    print("-----------------")
    
    # Support Vector Regression
    # SVR
    Cs = [0.1, 1, 10]
    epsilons = [0.1, 0.01, 0.001]
    for C, epsilon in product(Cs, epsilons):
        reg = SVR(C=C, epsilon=epsilon).fit(X_train, y_train)
        print(f"SVR coefficients {C=}, {epsilon=}:")
        print(f"Intercept: {reg.intercept_}")
        # print(f"Coefficients: {reg.coef_}")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        print("-----------------")

    # LinearSVR
    Cs = [0.1, 1, 10]
    epsilons = [0.1, 0.01, 0.001]
    for C, epsilon in product(Cs, epsilons):
        reg = LinearSVR(C=C, epsilon=epsilon).fit(X_train, y_train)
        print(f"LinearSVR coefficients {C=}, {epsilon=}:")
        print(f"Intercept: {reg.intercept_}")
        print(f"Coefficients: {reg.coef_}")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        print("-----------------")

    # NuSVR
    nus = [0.1, 0.5, 0.9]
    Cs = [0.1, 1, 10]
    for nu, C in product(nus, Cs):
        reg = NuSVR(nu=nu, C=C).fit(X_train, y_train)
        print(f"NuSVR coefficients {nu=}, {C=}:")
        print(f"Intercept: {reg.intercept_}")
        # print(f"Coefficients: {reg.coef_}")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        print("-----------------")

    # Decision Regression Tree
    max_depths = [2, 4, 6, 8, None]
    PATH_TREES = 'trees'
    os.makedirs(PATH_TREES, exist_ok=True)
    for max_depth in max_depths:
        reg = DecisionTreeRegressor(max_depth=max_depth).fit(X_train, y_train)
        print(f"Decision Tree coefficients {max_depth=}:")
        # print(f"Intercept: {reg.intercept_}")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        # plot tree
        print(f"Producing tree plot for max_depth={max_depth}...")
        plot_tree(reg, max_depth=5)
        plt.savefig(os.path.join(PATH_TREES, f"tree_max_depth_{max_depth}.png"), dpi=900)
        print("-----------------")

    # Random Forest
    n_estimators = [10, 20, 40]
    max_depths = [2, 4, None]
    PATH_FORESTS = 'forests'
    os.makedirs(PATH_FORESTS, exist_ok=True)
    for n_estimator, max_depth in product(n_estimators, max_depths):
        reg = RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth).fit(X_train, y_train)
        print(f"Random Forest coefficients {n_estimator=}, {max_depth=}:")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        # plot trees in each forest
        print(f"Producing tree plots for n_estimators={n_estimator}, max_depth={max_depth}...")
        os.makedirs(os.path.join(PATH_FORESTS, f"forest_n_estimators_{n_estimator}_max_depth_{max_depth}"), exist_ok=True)
        for i, tree in enumerate(reg.estimators_):
            plot_tree(tree, max_depth=5)
            plt.savefig(os.path.join(PATH_FORESTS, f"forest_n_estimators_{n_estimator}_max_depth_{max_depth}", f"tree_{i}.png"), dpi=900)
        print("-----------------")

    # Neural Network Regression (MLP)
    hidden_layer_sizes = [(10,), (20,), (10, 10), (20, 10), (10, 10, 10)]
    activations = ['relu', 'tanh']
    for hidden_layer_size, activation in product(hidden_layer_sizes, activations):
        reg = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation=activation).fit(X_train, y_train)
        print(f"Neural Network coefficients {hidden_layer_size=}, {activation=}:")
        # compute R^2
        print(f"R^2: {reg.score(X_test, y_test)}")
        print(f"MSE: {((y_test - reg.predict(X_test)) ** 2).mean()}")
        print(f"RMSE: {((y_test - reg.predict(X_test)) ** 2).mean() ** 0.5}")
        print(f"MAE: {abs(y_test - reg.predict(X_test)).mean()}")
        print("-----------------")
