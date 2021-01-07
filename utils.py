from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import KFold


def personal_normalisation(data):
    df = data.set_index('subject')
    for subj in df.index.unique():
        aux = df.loc[subj].iloc[:, :-1]
        cols = aux.columns
        aux = StandardScaler().fit_transform(aux)
        df.loc[subj, cols] = aux
    return df.reset_index()


def cv_leave_three_out(data, seed=123):
    train_indices = list()
    test_indices = list()

    subjects = data.subject.unique()
    np.random.seed(seed)
    np.random.shuffle(subjects)

    for i, subject in enumerate(subjects):
        testsubjects = [subjects[i], subjects[(i + 1) % len(subjects)], subjects[(i + 2) % len(subjects)]]
        trainsubjects = np.delete(subjects, [i, (i + 1) % len(subjects), (i + 2) % len(subjects)])

        test_i = list()
        for tests in testsubjects:
            test_i.append(np.array(data[data.subject == tests].index))
        test_indices.append(np.concatenate(test_i))

        train_i = list()
        for trains in trainsubjects:
            train_i.append(np.array(data[data.subject == trains].index))
        train_indices.append(np.concatenate(train_i))
    return train_indices, test_indices


def cv_loso(data):
    train_indices = list()
    test_indices = list()
    for subject in data.subject.unique():
        test_indices.append(np.array(data[data.subject == subject].index))
        train_indices.append(np.array(data[data.subject != subject].index))
    return train_indices, test_indices


def cv_indices_nfolds(data, **kwargs):
    """
    Return indices for leave n users out validation
    :param data: pandas df with column 'subject'
    :param seed: int, seed for reproducibility
    :param nfolds: int, number of folds
    :return: tuple of lists, train and test indices for each split
    """
    seed = kwargs['seed']
    if 'nfolds' in kwargs.keys():
        nfolds = kwargs['nfolds']
    else:
        nfolds = 8

    np.random.seed(seed)

    subjects = data.subject.unique()
    n_subject = len(subjects)
    kf = KFold(nfolds, random_state=seed, shuffle=True)
    splits = kf.split(np.arange(n_subject))

    train_indices = list()
    test_indices = list()
    n_test = list()
    for tr, te in splits:
        # break
        test_subjects = subjects[te]
        n_test.append(len(te))

        testinds = list()
        traininds = list()
        for subj in subjects:
            inds = np.where(subj == data.subject)[0]
            if subj in test_subjects:
                testinds.extend(list(inds))
            else:
                traininds.extend(list(inds))

        train_indices.append(traininds)
        test_indices.append(testinds)

    return train_indices, test_indices, n_test
