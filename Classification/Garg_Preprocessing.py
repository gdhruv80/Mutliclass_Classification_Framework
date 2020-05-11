import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


def train_test_dev_split(data_set, Output_col_name, dev_size=0.2, test_size=0.1):
    """
    :param data_set: Input DataFrame with all features
    :param Output_col_name:
    :param dev_size: Dev set is used for ROC curve operating point optimization
    :param test_size: Untouched test set used for reporting final numbers
    :return: X & Y's for 3 sets (Train dev and test)
    """
    # Creating a dev and test set
    tot_size = dev_size + test_size
    X, Y_ = data_set.loc[:, data_set.columns != Output_col_name], data_set[Output_col_name]

    X_train, X_S1, Y_train, Y_S1 = train_test_split(X, Y_, test_size=tot_size)
    X_dev, X_test, Y_dev, Y_test = train_test_split(X_S1, Y_S1, test_size=test_size / tot_size)
    return X_train, X_dev, X_test, Y_train, Y_dev, Y_test


def outlier_det(X, Y, X_plot_col, Y_plot_col, Output_mod='Isolation Forest', outliers_fraction=0.05):
    """
    :param X: DF (can include all cols except the one to be predicted)
    :param X_plot_col: X col for plotting outliers
    :param Y_plot_col: Y col for plotting outliers
    :param outliers_fraction: Percent of values to set as outliers
    :param Output_mod: Output model used for prediction ('Robust covariance' , 'Isolation Forest', 'Local Outlier
    Factor')
    :return: Outlier free version of Training DF (X_train , Y_train)
    """
    plt.figure(figsize=(15, 10))
    # Subsetting out rows with nan and categorical cols from outlier detection
    X_ = X[(X.isna().sum(axis=1)).apply(lambda x: False if x > 0 else True)]
    X_ = X_[X_.dtypes[X_.dtypes.isin([np.dtype('float64'), np.dtype('int64')])].index]

    anomaly_algorithms = [
        ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
        ("Isolation Forest", IsolationForest(behaviour='new', contamination=outliers_fraction, random_state=42)),
        ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))
    ]
    pred_dict = {}
    algo_cnt = 1
    for name, algorithm in anomaly_algorithms:
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X_)
        else:
            y_pred = algorithm.fit(X_).predict(X_)

        pred_dict[name] = y_pred
        # Plotting outliers
        plt.subplot(130 + algo_cnt)
        colors = np.array(['#377eb8', '#ff7f00'])
        labels = {-1: 'Outlier', 1: 'Non-Outlier'}
        for g in np.unique(y_pred):
            j = np.where(y_pred == g)
            plt.scatter(X_.iloc[j][X_plot_col], X_.iloc[j][Y_plot_col],
                        c=colors[(g + 1) // 2], label=labels[g])
        plt.xlabel(X_plot_col)
        plt.ylabel(Y_plot_col)
        plt.title('%s Outliers plotted along \n %s and %s'
                  % (name, X_plot_col, Y_plot_col,))
        plt.legend(loc="lower right")
        algo_cnt += 1

    # As this is multivariate outlier detection removing rows with outliers while predicting outliers using Output_mod
    y_pred = pred_dict[Output_mod]
    pred = pd.Series(y_pred, index=X_.index)
    outlier_index = pred[pred == -1].index
    X_outlier_free = X.drop(index=outlier_index)
    Y_outlier_free = Y.drop(index=outlier_index)
    return (X_outlier_free, Y_outlier_free)
