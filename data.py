import os
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

import neurokit2 as nk
from sklearn.ensemble import IsolationForest

from feature_extraction import get_all_hrv_features, eda_tonic_feature_extraction, eda_phasic_feature_extraction, \
    calculate_statistics

TEST_SUBJECTS = ['3caqi', '6frz4', 'bd47a', 'f1gjp', 'iz3x1']
TLX_COLS = ['TLX_mean', 'TLX_mental_demand', 'TLX_physical_demand', 'TLX_temporal_demand', 'TLX_performance',
            'TLX_effort', 'TLX_frustration']


def make_epochs(frame, wlen, overl):
    """
    Make epochs with length `wlen` and overlap `overl`
    :param frame: pandas dataframe with column `timestamp` which is time from beginning of session
    :param wlen: int, window length in seconds
    :param overl: int, overlap in seconds
    :return: list of epochs
    """

    frame = frame.copy()  # make sure not to modify original data
    # frame = df
    # wlen = 30
    # overl = 15

    ts_col = 'timestamp'

    # collect start and end times
    if frame[ts_col].max() < (frame[ts_col].min() + wlen + overl):
        starts = np.array([frame[ts_col].max() - wlen])
        ends = np.array([frame[ts_col].max()])
    else:
        if overl > 0:
            ends = np.sort(np.arange(frame[ts_col].max(), 0, -overl))
        else:
            ends = np.sort(np.arange(frame[ts_col].max(), 0, -wlen))
        ends = ends[ends >= wlen]
        starts = ends - wlen

    # create list of epochs
    if any(starts < 0):
        ends = ends[starts >= 0]
        starts = starts[starts >= 0]

    epochs = []
    for s, e in zip(starts, ends):
        epochs.append(frame[(frame[ts_col] > s) & (frame[ts_col] <= e)])

    return epochs


def physio_sanity(physio):
    """
    Sanity check for physiological data, a single value shouldn't be replicated too much
    :param physio: pandas dataframe
    :return: bool, True if the frame is ok
    """
    nuniq = np.array([physio.rr.nunique(), physio.hr.nunique(), physio.gsr.nunique()])
    lim = 0.25 * len(physio)
    if any(nuniq < lim):
        return False
    else:
        return True


def extract_physio_features(physio):
    """
    Actual workhorse function for extracting physiological features
    :param physio: pandas dataframe
    :return: pandas dataframe of physiological features
    """
    # ecg
    if physio_sanity(physio):

        correct_rrs = physio.rr.values * 1000

        # get basic statistics
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        hr = calculate_statistics(physio.hr.values, 'hr')
        rr = calculate_statistics(correct_rrs, 'rr')
        gsr = calculate_statistics(physio.gsr.values, 'gsr')
        temp = calculate_statistics(physio.temperature.values, 'temp')

        # get hrv variables
        warnings.filterwarnings("ignore", category=UserWarning)
        hrv = get_all_hrv_features(correct_rrs)
        hrv = hrv.drop(['mean_hr', 'max_hr', 'min_hr', 'std_hr', 'tinn'], axis=1)
        hrv.columns = ['hrv__' + c for c in hrv.columns]
        warnings.filterwarnings("default", category=UserWarning)

        # get eda features
        eda = physio.gsr
        eda = eda.rolling(3).mean().bfill().values
        eda_decomposed = nk.eda_phasic(eda, sampling_rate=1, method='cvxeda')
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            peak_signal, info = nk.eda_peaks(eda_decomposed["EDA_Phasic"].values, sampling_rate=1,
                                             method='neurokit', amplitude_min=0.2)
        except ValueError:  # if there are no peaks found
            sig_cols = ['SCR_Onsets', 'SCR_Peaks', 'SCR_Height', 'SCR_Amplitude',
                        'SCR_RiseTime', 'SCR_Recovery', 'SCR_RecoveryTime']
            peak_signal = pd.DataFrame(np.zeros(7*len(eda)).reshape(len(eda), 7), columns=sig_cols)
            info = {s: [np.nan] for s in sig_cols}

        warnings.filterwarnings("default", category=RuntimeWarning)

        # combine decomposed signals and peak detection results
        signals = pd.DataFrame({"EDA_Raw": physio.gsr.values, 'EDA_Clean': eda})
        signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)
        # nk.eda_plot(signals, 1)

        # get features for phasic and tonic component
        phasic = eda_phasic_feature_extraction(signals)
        tonic = eda_tonic_feature_extraction(signals.EDA_Tonic.values)
        tonic = pd.DataFrame.from_dict(tonic, orient='index').T
        eda = pd.concat((phasic, tonic), axis=1)

        return pd.concat((hr, hrv, rr, gsr, eda, temp), axis=1)
    else:
        return None


def get_feats_session(df, wlen, overl):
    """
    Calculate features for a single session
    :param df: pandas dataframe with single session data
    :param wlen: int, window length in seconds
    :param overl: int, overlap in seconds
    :return: pandas dataframe of features, or None, if none could be extracted
    """

    # convert timestamp to seconds from beginning
    df['datetime'] = pd.to_datetime(df.datetime)
    df['timestamp'] = (df.datetime - df.datetime.min()).dt.total_seconds().round(0)

    # turn data into a list of epochs
    epochs = make_epochs(df, wlen, overl)

    # obtain window-wise features
    feats = list()
    for epoch in epochs:
        # break
        if len(epoch) >= wlen:
            if int(epoch.timestamp.max() + 1 - epoch.timestamp.min()) == len(epoch):
                ff = extract_physio_features(epoch)
                if ff is not None:
                    ff = pd.concat((ff, epoch[TLX_COLS].iloc[:1].reset_index(drop=True)), axis=1)
                feats.append(ff)

    # if features could be extracted, combine
    if (len(feats) > 0) and (np.sum([f is None for f in feats]) < len(feats)):
        feats = pd.concat(feats)
        feats = feats.reset_index(drop=True)
        feats.index.name = 'win'
    else:
        feats = None

    return feats


def rename_rest_tasks(df):
    """
    Helper function to rename the resting tasks to contain identifier of previous task
    :param df: pandas dataframe
    :return: pandas dataframe
    """

    # get indices where level changes to `rest`
    lvl = df.level.values
    inds = np.where(lvl == 'rest')[0]
    inds = [i for i in inds if i-1 not in inds]

    # go through resting indices and append the level of previous task
    for ind in inds:
        # break
        toapp = lvl[ind - 1]
        while lvl[ind] == 'rest':
            lvl[ind] = 'rest' + toapp
            ind += 1

    df['level'] = lvl
    return df


def calculate_features(path, wlen, overl):
    """
    Actual workhorse function to calculating features for all users
    :param path: string, path to directory containing each users data
    :param wlen: int, window length in seconds
    :param overl: int, overlap in seconds
    """

    # path = './data/train/'
    # wlen = 30
    # overl = 15

    # go through the files in path
    feature_data = []
    for file in os.listdir(path):
        # break
        dat = pd.read_csv(path + file)  # read in data
        dat = rename_rest_tasks(dat)  # rename tasks
        dat = dat[(dat.task != 'quest') & (dat.task != 'post')]  # remove unneeded tasks

        # for ind in dat.set_index(['task', 'level']).index.unique():
        #     ind = ('HP', '0')
        #     break
        #     df = dat.set_index(['task', 'level']).loc[ind]
        #     # if len(df) < wlen:
        #     #     break
        #     ff = get_feats_session(df, wlen, overl)

        # df = dat[(dat.task == 'GCrest') & (dat.level == 'rest0')]

        # calculate features and append to the list
        feats = dat.groupby(['task', 'level']).apply(lambda x: get_feats_session(x, wlen, overl))
        feats = feats.reset_index()
        feats.insert(0, 'subject', file.split('_')[0])
        feature_data.append(feats)
        # print(file)

    # combine
    feats = pd.concat(feature_data)

    # remove ill-defined variables
    # feats = feats.dropna(axis=1)  # remove columns with missing values
    # feats = feats.set_index(['subject', 'task', 'level', 'win'])
    # feats = feats[feats.columns[np.isinf(feats).sum() == 0]]  # remove columns with infinite values
    # feats = feats[feats.columns[feats.var() >= 0.01]]  # remove constants and almost constants

    # store to disk
    overl_str = str(overl).replace('.', '')
    feats.to_csv(f'./data/features_{wlen}s_{overl_str}s.csv', index=False)


def get_data_user(file):
    """
    Read in a single user's data with added subject column
    :param file: string, path to file
    :return: pandas dataframe
    """
    df = pd.read_csv(file)
    df['subject'] = file.split('_')[0].split('/')[-1]
    return df


def map_level_to_classification(df):
    """
    Make a response variable for classification
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    mapper = {'0': 1, '1': 1, '2': 1}  # actual tasks all to class one
    rest_mapper = {'rest0': 0, 'rest1': 0, 'rest2': 0}  # rest tasks class zero
    mapper.update(rest_mapper)
    lvl = df.level.map(mapper).to_frame()
    df['y'] = lvl
    return df


def read_features(wlen, overl, path='./data/', preprocess=False):
    """
    Read in features table for given settings
    :param wlen: int, window length used for feature calculation in seconds
    :param overl: int, overlap used for feature calculation in seconds
    :param path: string, path to folder containing the data
    :param preprocess: boolean, whether to preprocess the data
    :return: pandas dataframe
    """
    overl_str = str(overl).replace('.', '')
    df = pd.read_csv(f'{path}features_{wlen}s_{overl_str}s.csv')
    df = map_level_to_classification(df)
    if preprocess:
        df = drop_ill_defined(df)
        df = map_level_to_classification(df)
        df = df.dropna()
        df = df.reset_index(drop=True)
        # ind = df[['task', 'level', 'win']]
        # df = df.drop(['task', 'level', 'win'], axis=1)
        # df = df.set_index(['task', 'level', 'win'])
        # df = personal_normalisation(df)
        # df = pd.concat((ind, df), axis=1)
        #
        # df = drop_outliers(df)
        # df = df.reset_index(drop=True)

    return df


def drop_ill_defined(feats):
    feats = feats.dropna(axis=1)  # remove columns with missing values
    feats = feats.set_index(['subject', 'task', 'level', 'win'])
    feats = feats[feats.columns[np.isinf(feats).sum() == 0]]  # remove columns with infinite values
    feats = feats[feats.columns[feats.var() >= 0.01]]  # remove constants and almost constants
    return feats.reset_index()


def drop_outliers(df):
    inds = [i for i in range(len(df)) if df.subject.iloc[i] in TEST_SUBJECTS]
    test = df.iloc[inds]
    train = df.drop(inds)

    xtrain = train.drop(['subject', 'task', 'level', 'win', 'y'], axis=1)
    xtest = test.drop(['subject', 'task', 'level', 'win', 'y'], axis=1)

    isof = IsolationForest()
    isof.fit(xtrain)

    preds = isof.predict(xtrain)
    inds = np.where(preds == 1)[0]
    train = train.iloc[inds]

    preds = isof.predict(xtest)
    inds = np.where(preds == 1)[0]
    test = test.iloc[inds]

    return pd.concat((train, test))


if __name__ == '__main__':
    # wlen = 30
    # overl = 15
    # path = './data/'

    path = './data/train/'

    calculate_features(path, 30, 15)
    print(datetime.now())
    calculate_features(path, 25, 12.5)
    print(datetime.now())
    calculate_features(path, 20, 10)
    print(datetime.now())
    calculate_features(path, 15, 7.5)
    print(datetime.now())
    calculate_features(path, 10, 5)
    print(datetime.now())
    calculate_features(path, 5, 2.5)
    print(datetime.now())

