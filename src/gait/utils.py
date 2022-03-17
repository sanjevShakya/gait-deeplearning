import os
import tensorflow as tf
from gait.config import pd
from gait.config import np
from gait.constants import ROOT_DATA_DIR, SUBJECT_FILE, Y_FILE, X_PATH, X_LABELS
from numpy import genfromtxt

SENSORS = {
    "LEFT": "LEFT",
    "RIGHT": "RIGHT",
}
SENSORS_LIST = [SENSORS["LEFT"], SENSORS["RIGHT"]]
sessions = ['session1', 'session2', 'session3', 'session4', 'session5', 'session6']
# sessions = ['session4']
DEFAULT_SESSIONS = sessions[0]


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
    dataframe = pd.read_csv(filename, header=None,
                            delimiter=",")
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


def path_builder(session, overlapPercent, sensorName, fileName, prefix=""):
    return ROOT_DATA_DIR + session + '/' + sensorName + "/" + get_data_overlap_folder(overlapPercent) + '/' + prefix + fileName


def get_unique_subjects(subjects):
    return np.unique(subjects)


def remove_invalid_data(X, y, subjects):
    nan_indexes = np.argwhere(np.isnan(y))[:, 0]
    if nan_indexes.size != 0:
        y = np.delete(y, nan_indexes[0], axis=0)
        subjects = np.delete(subjects, nan_indexes[0], axis=0)
        X = np.delete(X, nan_indexes[0], axis=0)
    return X, y, subjects


def get_overlap_data_all_sessions(overlapPercent, xLabels=X_LABELS):
    X_list = list()
    y_list = list()
    subject_list = list()
    for session in sessions:
        X, y, subject = get_data_by_overlap_percent(
            overlapPercent, xLabels=X_LABELS, session=session)
        X_list.append(X)
        y_list.append(y)
        subject_list.append(subject)

    return np.vstack(X_list), np.vstack(y_list), np.vstack(subject_list)


def get_data_by_overlap_percent(overlapPercent, xLabels=X_LABELS, session=DEFAULT_SESSIONS):

    subject_file_path_left = path_builder(session,
                                          overlapPercent, SENSORS["LEFT"], SUBJECT_FILE)
    y_file_path_left = path_builder(
        session, overlapPercent, SENSORS["LEFT"],  Y_FILE)
    x_files = list(map(lambda label: get_X_files(label), xLabels))
    X_files_path_left = list(
        map(lambda fileName: path_builder(session, overlapPercent, SENSORS["LEFT"], fileName, prefix=X_PATH), x_files))
    X_left = load_group(X_files_path_left)
    y_left = load_file(y_file_path_left)
    subject_left = load_file(subject_file_path_left)

    subject_file_path_right = path_builder(session,
                                           overlapPercent, SENSORS["RIGHT"], SUBJECT_FILE)
    y_file_path_right = path_builder(
        session, overlapPercent, SENSORS["RIGHT"],  Y_FILE)
    x_files = list(map(lambda label: get_X_files(label), xLabels))
    X_files_path_right = list(
        map(lambda fileName: path_builder(session, overlapPercent, SENSORS["RIGHT"], fileName, prefix=X_PATH), x_files))
    X_right = load_group(X_files_path_right)
    y_right = load_file(y_file_path_right)
    subject_right = load_file(subject_file_path_right)
    X = np.concatenate((X_left, X_right), axis=0)
    y = np.concatenate((y_left, y_right), axis=0)
    subject = np.concatenate((subject_left, subject_right), axis=0)

    X, y, subject = remove_invalid_data(X, y, subject)
    y = np.array(y, dtype=float)
    y = np.array(y, dtype=int)
    y = np.array(y, dtype=str)

    subject = np.array(subject, dtype=str)
    return (X, y, subject)


def filter_excluded_subject(subjects, excluded_subjects):
    return [subject for subject in subjects if subject not in excluded_subjects]


def split_test_train_by_subjects(X, y, subjects, train_percent=0.8, exclude_subjects=[]):
    unique_subjects = get_unique_subjects(subjects)
    unique_subjects = filter_excluded_subject(
        unique_subjects, exclude_subjects)
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
    train_y = np.array(train_y, dtype=float)
    test_y = np.array(test_y, dtype=float)
    train_y = np.array(train_y, dtype=int)
    test_y = np.array(test_y, dtype=int)

    train_y = train_y
    test_y = test_y
    encoded_train_y = tf.keras.utils.to_categorical(train_y)
    encoded_test_y = tf.keras.utils.to_categorical(test_y)

    return train_X, test_X, encoded_train_y, encoded_test_y, train_y, test_y
