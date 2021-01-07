import os
import numpy as np
import pickle
import argparse
from statsmodels.stats.weightstats import DescrStatsW

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

from utils import personal_normalisation, cv_indices_nfolds
from data import read_features

import csv
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
from timeit import default_timer as timer

MODALITIES = ['gsr', 'hr', 'rr', 'temp']
TEST_SUBJECTS = ['3caqi', '6frz4', 'bd47a', 'f1gjp', 'iz3x1']
USE_TEST_SUBJECTS = False


parser = argparse.ArgumentParser()
parser.add_argument('--wlen', action='store', default='30', choices=['5', '10', '15', '20', '25', '30'],
                    help="Window length in seconds")
parser.add_argument('--overlap', action='store', default='15', choices=['2.5', '5', '7.5', '10', '12.5', '15'],
                    help="Overlap in seconds")
parser.add_argument('--data_path', action='store', default='./data/', help="Path to feature data")
parser.add_argument('--results_path', action='store', default='./results/', help="Path to results folder")
# opts = parser.parse_args('--wlen 30 --overlap 15'.split(' '))
opts = parser.parse_args()


if not os.path.exists(opts.results_path):
    os.makedirs(opts.results_path)

ITERATION = 0

overl_str = str(opts.overlap).replace('.', '')
identifier = f'{opts.wlen}s_{overl_str}s'

FILENAME = f'./{opts.results_path}/xgb_bayes_opt_{identifier}.csv'
TRIALS_NAME = f'./{opts.results_path}/xgb_bayes_trials_{identifier}.pickle'
MAX_EVALS = 50
SEED = 36147

print(opts)


def objective(params):
    """Objective function for XGB Hyperparameter Optimization"""

    # Keep track of evals
    global ITERATION

    ITERATION += 1

    start = timer()

    # check that two params are integers
    params['n_estimators'] = int(params['n_estimators'])
    params['max_depth'] = int(params['max_depth'])
    # params['n_estimators'] = 2
    # params['max_depth'] = 1

    # define estimator
    xgb = XGBClassifier(random_state=SEED, **params)
    # xgb.set_params(**params)

    # Perform n_folds cross validation
    scores_list = list()
    avg_score = list()
    avg_std_score = list()
    n_test_users = list()
    test_scores = list()
    for tr, te in cv:
        # break
        x_train_i = X_train.iloc[tr]
        x_train_i = x_train_i.reset_index(drop=True)
        y_tr_i = y_train.iloc[tr].values

        # inner cv indeces
        # data = train.iloc[tr].reset_index(drop=True)
        # kwargs = {'seed': SEED, 'nfolds': 5}

        tr_i, val_i, ntest_i = cv_indices_nfolds(train.iloc[tr].reset_index(drop=True), seed=SEED, nfolds=10)
        # print(ntest_i)

        # cross-validate
        scores = cross_val_score(xgb, x_train_i, y_tr_i, scoring='accuracy', cv=zip(tr_i, val_i), n_jobs=-1)
        wstat = DescrStatsW(scores, weights=ntest_i, ddof=1)
        avg = wstat.mean
        avg_std = wstat.std

        scores_list.append(scores)
        avg_score.append(avg)
        avg_std_score.append(avg_std)
        n_test_users.append(ntest_i)

        # monitor test scores
        xgb.fit(x_train_i, y_tr_i)
        te_pred = xgb.predict(X_train.iloc[te])
        test_scores.append(accuracy_score(y_train.iloc[te].values, te_pred))

    # acc = np.mean(scores_list, axis=1).mean()  # validation acc - unweighted
    wstat = DescrStatsW(avg_score, weights=len(train.subject.unique()) - np.array(ntest), ddof=1)
    val_acc = wstat.mean  # validation acc
    val_acc_std = wstat.std  # std of avg score
    # val_acc_std = np.average(avg_std_score, weights=len(train.subject.unique()) - np.array(ntest))  # validation acc
    wstat = DescrStatsW(test_scores, weights=ntest, ddof=1)
    test_val_acc = wstat.mean  # mean test score
    test_val_acc_std = wstat.std  # std of test score

    run_time = timer() - start

    # Loss must be minimized
    loss = 1 - val_acc

    # Write to the csv file
    out_file = os.getcwd() + os.sep + FILENAME
    of_connection = open(out_file, 'a')
    writer = csv.writer(of_connection)
    if USE_TEST_SUBJECTS:
        xgb.fit(X_train, y_train)
        test_acc = accuracy_score(y_test, xgb.predict(X_test))  # monitor test acc just for fun
        writer.writerow([val_acc,  val_acc_std, test_val_acc, test_acc, params, ITERATION, run_time])
    else:
        test_acc = np.nan
        writer.writerow([val_acc, val_acc_std, test_val_acc, test_val_acc_std, test_acc, params, ITERATION, run_time])
    of_connection.close()

    # Dictionary with information for evaluation
    return {'loss': loss, 'params': params, 'iteration': ITERATION,
            'train_time': run_time, 'status': STATUS_OK}


if __name__ == '__main__':
    # get feature data and normalise
    df = read_features(opts.wlen, opts.overlap, opts.data_path, True)
    df = df.drop(['task', 'level', 'win'], axis=1)
    df = df.drop([c for c in df.columns if 'TLX' in c], axis=1)
    df = personal_normalisation(df)

    # split into training/validation and testing
    if USE_TEST_SUBJECTS:
        inds = [i for i in range(len(df)) if df.subject.iloc[i] in TEST_SUBJECTS]
        test = df.iloc[inds]
        train = df.drop(inds)
        train = train.reset_index(drop=True)
        test = test.reset_index(drop=True)

        X_train = train.drop(['subject', 'y'], axis=1)
        y_train = train.y

        X_test = test.drop(['subject', 'y'], axis=1)
        y_test = test.y
    else:
        train = df.reset_index(drop=True)
        X_train = train.drop(['subject', 'y'], axis=1)
        y_train = train.y

    # obtain outer cv indeces
    # trainind, testind = cv_leave_three_out(train, seed=SEED)
    trainind, testind, ntest = cv_indices_nfolds(train, seed=SEED, nfolds=11)
    cv = list(zip(trainind, testind))

    # define search space
    space = {
        # 'boosting': hp.choice('boosting', ['gbtree', 'gblinear', 'dart']),
        'n_estimators': hp.quniform('n_estimators-xg', 20, 250, 10),
        'max_depth': hp.quniform('max_depth-xg', 2, 12, 1),
        'learning_rate': hp.uniform('learning_rate-xg', 0.01, 0.5),
        'gamma': hp.choice('gamma', [0, hp.uniform('gamma-xg', 0, 0.05)]),
        'subsample': hp.choice('subsample', [1, hp.uniform('subsample-xg', 0.7, 1)]),
        'colsample_bytree': hp.choice('colsample_bytree', [1, hp.uniform('colsample_bytree-xg', 0.7, 1)]),
        'colsample_bynode': hp.choice('colsample_bynode', [1, hp.uniform('colsample_bynode-xg', 0.7, 1)]),
        'colsample_bylevel': hp.choice('colsample_bylevel', [1, hp.uniform('colsample_bylevel-xg', 0.7, 1)]),
        'reg_alpha': hp.uniform('reg_alpha-xg', 0, 1),
        'reg_lambda': hp.uniform('reg_lambda-xg', 0, 1)}

    # from hyperopt.pyll.stochastic import sample
    # params = sample(space)

    tpe_algorithm = tpe.suggest
    trials_obj = TRIALS_NAME
    multip = 1
    if os.path.isfile(trials_obj):
        print("Reading in old trials file")
        with open(trials_obj, 'rb') as fp:
            bayes_trials = pickle.load(fp)
        iters = bayes_trials.results[-1]['iteration']
        MAX_EVALS += iters
        ITERATION += iters
    else:
        bayes_trials = Trials()

    # File to save results
    out_file = os.getcwd() + os.sep + FILENAME
    if not os.path.isfile(out_file):
        of_connection = open(out_file, 'w')
        writer = csv.writer(of_connection)

        # Write the headers to the file
        writer.writerow(['val_acc', 'val_acc_std', 'test_val_acc', 'test_val_acc_std', 'test_acc', 'params',
                         'iteration', 'validation_time'])
        of_connection.close()
    else:
        print("Appending to old results file")

    # Run optimization
    try:
        best = fmin(fn=objective, space=space, algo=tpe.suggest,
                    max_evals=MAX_EVALS, trials=bayes_trials)
    except KeyboardInterrupt:
        with open(trials_obj, 'wb') as fp:
            pickle.dump(bayes_trials, fp)

    with open(trials_obj, 'wb') as fp:
        pickle.dump(bayes_trials, fp)


# import pandas as pd
# res = pd.read_csv('./results/xgb_bayes_opt_30s_15s.csv')
# res.sort_values('acc')
