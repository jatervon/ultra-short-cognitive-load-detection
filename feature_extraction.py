from hrvanalysis import get_time_domain_features, get_csi_cvi_features, get_frequency_domain_features, \
    get_geometrical_features, get_poincare_plot_features
import pandas as pd
import numpy as np
from scipy.integrate import simps


HRV_FUNCTIONS = [get_time_domain_features, get_csi_cvi_features,
                 lambda x: get_frequency_domain_features(x, sampling_frequency=1),
                 get_geometrical_features, get_poincare_plot_features]


def calculate_statistics(arr, name):
    """
    Calculate basic statistics from a 1d signal
    :param arr: numpy array, the signal whose statistics will be calculated
    :param name: string, identifier for the signal name
    :return: pandas dataframe of features
    """
    def iqr(x):
        """
        Compute interquartile length
        :param x: numpy array
        :return: np.float64
        """
        return np.quantile(x, .75) - np.quantile(x, .25)

    def var_range(x):
        """
        Calculate the range of the variable
        :param x: numpy array
        :return: np.float64
        """
        return np.max(x) - np.min(x)

    # collect functions into a list
    funcs = [np.mean, np.std, np.min, np.max, lambda x: np.quantile(x, .25), lambda x: np.quantile(x, .75), iqr,
             var_range, lambda x: np.std(x) / np.mean(x)]

    # compute stats for the signal and its 1st and 2nd derivative
    stats = [f(arr) for f in funcs]
    stats_diff1 = [f(np.diff(arr)) for f in funcs]
    stats_diff2 = [f(np.diff(arr, 2)) for f in funcs]
    feats = stats + stats_diff1 + stats_diff2

    # naming
    f_ids = ['d0', 'd1', 'd2']
    names = ['mean', 'std', 'min', 'max', 'lq', 'uq', 'iqr', 'range', 'cv']
    feat_names = [[name + '__' + f + '_' + n for n in names] for f in f_ids]
    feat_names = np.concatenate(feat_names)

    return pd.DataFrame.from_dict({feat_names[i]: feats[i] for i in range(len(feats))}, orient='index').T


def get_all_hrv_features(ibis):
    """
    Wrapper to calculate hrv measures
    :param ibis: numpy array containing interbeat intervals (in ms)
    :return: pandas dataframe of hrv features
    """
    feats = [func(ibis) for func in HRV_FUNCTIONS]
    return pd.concat([pd.DataFrame.from_dict(feat, orient='index').T for feat in feats], axis=1)


def eda_tonic_feature_extraction(tonic):
    """
    Computes SCL features
    :param tonic: numpy array, the tonic component of eda signal
    :return: dict of scl features
    """
    feats = list()
    # basic stats
    feats.extend(list(pd.Series(tonic).describe().loc[['mean', 'std', 'min', 'max']].values))
    feats.append(feats[-1] / feats[-2])  # slope
    feats.append(np.corrcoef(tonic, np.arange(len(tonic)))[1, 0])  # correlation with time

    # naming
    names = ['mean', 'std', 'min', 'max', 'slope', 'corrwithtime']
    names = ['eda__scl_' + c for c in names]
    return {names[i]: feats[i] for i in range(len(names))}


def eda_phasic_feature_extraction(eda_processed):
    """
        Compute a simple set of SCR features, those available even if only partial/no SCRs occur during the window
        :param eda_processed: pandas df, the ´bio_df´ as returned by neurokit2::bio_process
        :return: dict of phasic EDA features
        """

    # number of peaks
    npeaks = eda_processed['SCR_Peaks'].values.sum()
    npeaks = pd.Series(int(npeaks), index=['npeaks'])

    # basic statistics of first and second derivative
    diff1 = eda_processed.EDA_Phasic.diff().describe().drop(['count'])
    diff2 = eda_processed.EDA_Phasic.diff(2).describe().drop(['count'])
    diff1.index = ['diff1_' + s.replace('50%', 'median').replace('25%', 'lq').replace('75%', 'uq') for s in diff1.index]
    diff2.index = ['diff2_' + s.replace('50%', 'median').replace('25%', 'lq').replace('75%', 'uq') for s in diff2.index]

    # total time of increase and decrease
    risetime = (eda_processed.EDA_Phasic.diff() > 0).sum() / len(eda_processed)
    rectime = (eda_processed.EDA_Phasic.diff() < 0).sum() / len(eda_processed)
    time = pd.Series((risetime, rectime), index=['risetime', 'dectime'])
    ret = pd.DataFrame(pd.concat((npeaks, diff1, diff2, time))).T
    ret.columns = ['eda__phasic_' + c for c in ret.columns]

    return ret


def eda_scr_feature_extraction(eda_processed, eda_info):
    """
    Compute SCR features
    :param eda_processed: pandas df, the ´bio_df´ as returned by neurokit2::bio_process
    :param eda_info: dict, the ´bio_info´ as returned by neurokit2::bio_process
    :return: dict of SCR features
    """
    feats = list()

    # get mean, std, number of peaks
    feats.extend([eda_processed.EDA_Phasic.mean(), eda_processed.EDA_Phasic.std()])
    feats.append(eda_processed['SCR_Peaks'].values.sum())
    names = ['mean', 'std', 'npeaks']

    # retrieve scr areas
    areas = list()
    for o, onset in enumerate(eda_info['SCR_Onsets']):
        offset = eda_info['SCR_Recovery'][o]

        if np.isnan(onset) | np.isnan(offset):
            areas.append(np.nan)
        else:
            onset = int(onset)
            offset = int(offset)
            areas.append(simps(eda_processed.EDA_Phasic.iloc[onset:offset]))

    areas = np.array(areas)
    eda_info['SCR_Areas'] = areas

    # retrieve features describing shapes of peaks
    def scr_shape_feats(aspect):
        auxer = pd.Series(eda_info[aspect])
        fs = list(auxer.describe().loc[['mean', 'std', 'min', 'max']].values)
        fs.append(np.nansum(auxer))
        return tuple(fs)

    feats.extend(scr_shape_feats('SCR_Amplitude'))
    feats.extend(scr_shape_feats('SCR_Height'))
    feats.extend(scr_shape_feats('SCR_RiseTime'))
    feats.extend(scr_shape_feats('SCR_RecoveryTime'))
    feats.extend(scr_shape_feats('SCR_Areas'))

    # feature names
    names.extend(['ampli_mean', 'ampli_std', 'ampli_min', 'ampli_max', 'ampli_sum'])
    names.extend(['height_mean', 'height_std', 'height_min', 'height_max', 'height_sum'])
    names.extend(['risemean', 'risestd', 'risemin', 'risemax', 'risesum'])
    names.extend(['recmean', 'recstd', 'recmin', 'recmax', 'recsum'])
    names.extend(['areamean', 'areastd', 'areamin', 'areamax', 'areasum'])
    names = ['eda__scr_' + c for c in names]

    # convert to dict and return
    return {names[i]: [feats[i]] for i in range(len(names))}

