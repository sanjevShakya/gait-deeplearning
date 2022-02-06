import os
import tensorflow as tf
from gait.config import pd
from gait.config import np
from gait.constants import ROOT_DATA_DIR, SUBJECT_FILE, Y_FILE, X_PATH, X_LABELS


def get_X_files(label):
    '''
    returns X data file names
    '''
    return 'acc_{}_data.csv'.format(label)


def get_data_overlap_folder(overlapPercent):
    '''
    returns overlapping data foldername
    '''
    return 'data_{}_overlap'.format(overlapPercent)


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def load_file(filename):
    '''
    load data from a filename
    '''
    dataframe = pd.read_csv(filename, header=None, delimiter=",")
    return dataframe.values


def load_group(filenames):
    '''
    load data from a list of filenames
    '''
    loaded = list()
    for name in filenames:
        data = load_file(name)
        loaded.append(data)
    loaded = np.dstack(loaded)
    return loaded


def path_builder(overlapPercent, fileName, prefix=""):
    return ROOT_DATA_DIR + get_data_overlap_folder(overlapPercent) + '/' + prefix + fileName


def get_unique_subjects(subjects):
    return np.unique(subjects)


def get_data_by_overlap_percent(overlapPercent):
    subject_file_path = path_builder(overlapPercent, SUBJECT_FILE)
    y_file_path = path_builder(overlapPercent, Y_FILE)
    x_files = list(map(lambda label: get_X_files(label), X_LABELS))
    X_files_path = list(
        map(lambda fileName: path_builder(overlapPercent, fileName, prefix=X_PATH), x_files))
    X = load_group(X_files_path)
    y = load_file(y_file_path)
    subject = load_file(subject_file_path)

    return (X, y, subject)


def split_test_train_by_subjects(X, y, subjects, train_percent=0.8):
    unique_subjects = get_unique_subjects(subjects)
    np.random.shuffle(unique_subjects)
    M = len(unique_subjects)
    m_train = int(M * train_percent)
    train_subjects = unique_subjects[0:m_train]
    test_subjects = unique_subjects[m_train:M+1]
    train_idx = np.where(subjects == train_subjects)[0]
    test_idx = np.where(subjects == test_subjects)[0]

    train_X = X[train_idx, :]
    test_X = X[test_idx, :]
    train_y = y[train_idx, :]
    test_y = y[test_idx, :]

    train_y = train_y - 1
    test_y = test_y - 1
    encoded_train_y = tf.keras.utils.to_categorical(train_y)
    encoded_test_y = tf.keras.utils.to_categorical(test_y)

    return train_X, test_X, encoded_train_y, encoded_test_y, train_y, test_y
