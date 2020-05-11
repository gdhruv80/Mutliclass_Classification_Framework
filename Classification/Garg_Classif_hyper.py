import numpy as np
from numpy.random import uniform as uni, randint as unint
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.impute import IterativeImputer
from sklearn.linear_model import SGDClassifier, BayesianRidge
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

# General functions
roundnp = np.vectorize(lambda t: round(t, 3))


# Utility function to report best scores from hyper parameter search
def report_top_hyp(results, scoring_crit, n_top=3):
    """
    :param results: cv_results_
    :param n_top: no of top results to see
    :param scoring_crit: AUC or TPR, FPR or other scoring criteria
    :return: None
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation %s : %.4f  (std: %.2f)" % (scoring_crit, results['mean_test_score'][candidate],
                                                              results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# Feature and Model parameter options for creating the pipeline
def param_grid(feat_selec, mod_type, hyper_selc, rand_search_iter):
    # Look into making PCA, Kbest and l1 dynamic
    feat_param = {
        'PCA': {'PCA__n_components': [5, 20, 300] if hyper_selc == 'Grid' else unint(5, 30, rand_search_iter)},
        'KBest': {'KBest__k': [5, 20, 30] if hyper_selc == 'Grid' else unint(5, 30, rand_search_iter)},
        'L1': {'L1__max_features': [5, 20, 30] if hyper_selc == 'Grid' else unint(5, 30, rand_search_iter)}}

    mod_param = {'gbc': {'gbc__learning_rate': [0.1, 0.005, 0.001],
                         'gbc__n_estimators': [50, 100, 200],
                         'gbc__min_samples_split': [0.1, 0.3, 0.6, 0.8]},
                 'rf': {'rf__max_depth': [5, 8, 12, 15, 25] if hyper_selc == 'Grid' else unint(4, 25, rand_search_iter),
                        'rf__n_estimators': [30, 50, 80, 100, 130] if hyper_selc == 'Grid' else unint(30, 150,
                                                                                                      rand_search_iter),
                        'rf__max_features': [0.2, 0.5, 0.8] if hyper_selc == 'Grid' else roundnp(
                            uni(0.1, 0, rand_search_iter)),
                        'rf__min_samples_split': [2, 8, 10, 15, 20, 30] if hyper_selc == 'Grid' else unint(2, 30,
                                                                                                           rand_search_iter)},
                 'lr': {'lr__penalty': ['l2', 'l1', 'elasticnet'],
                        'lr__alpha': [0.0001, 0.0002, 0.0005],
                        'lr__max_iter': [3, 5, 7, 9]},
                 'svm': {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000] if hyper_selc == 'Grid' else roundnp(
                     uni(0.0001, 1000, rand_search_iter)),
                         'svm__gamma': [0.01, 0.1, 1, 10, 100] if hyper_selc == 'Grid' else roundnp(
                             uni(0.0001, 1000, rand_search_iter)),
                         'svm__kernel': ['linear', 'poly', 'rbf']}
                 }
    final_parm = feat_param[feat_selec]
    final_parm.update(mod_param[mod_type])
    return final_parm


def imputers(mod_type, rs):
    mods = {
        'bays': BayesianRidge(),
        'Rf': RandomForestRegressor(max_features='sqrt', min_samples_split=10, min_samples_leaf=5, random_state=rs),
        'Knn': KNeighborsRegressor(n_neighbors=15)
        # ExtraTreesRegressor(n_estimators=10, random_state=0),
    }
    try:
        return IterativeImputer(random_state=rs, estimator=mods[mod_type])
    except KeyError:
        raise Exception('Model selection method not found in the list. Please select either '
                        'Tree based or Statistical based methods.')


# Model config to use for creating the pipeline
def models(mod_type, rs):
    gbc = GradientBoostingClassifier(random_state=rs)
    rf = RandomForestClassifier(random_state=rs)
    lr = SGDClassifier(loss='log', random_state=rs)
    svm = SVC(random_state=rs)
    mods = {'Tree_based': {'gbc': gbc, 'rf': rf}, 'Stat_based': {'lr': lr, 'svm': svm}}
    try:
        return mods[mod_type]
    except KeyError:
        raise Exception('Model selection method not found in the list. Please select either '
                        'Tree based or Statistical based methods.')


def feature_selection(method):
    pca = PCA()
    L1 = SelectFromModel(LinearSVC(penalty="l1", dual=False, max_iter=5000), threshold=-np.inf)
    KBest = SelectKBest()
    methods = {'PCA': pca, 'L1': L1, 'KBest': KBest}
    try:
        return methods[method]
    except KeyError:
        raise Exception('Feature selection method not found in the list')


def hyperparam_search(method, pipe, param_grid, rand_search_iter, scoring_crit):
    Grid = GridSearchCV(pipe, param_grid, iid=True,
                        cv=5, return_train_score=False, n_jobs=-1, scoring=scoring_crit)
    Random = RandomizedSearchCV(pipe, param_distributions=param_grid, iid=True,
                                cv=5, return_train_score=False, n_jobs=-1, n_iter=rand_search_iter,
                                scoring=scoring_crit)
    methods = {'Grid': Grid, 'Random': Random}
    try:
        return methods[method]
    except KeyError:
        raise Exception('Hyperparameter selection method not found in the list. Please select either '
                        'Randomized or Grid search based methods.')
