# Path relative to notebook

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent


ROOT_DATA_DIR = '../data/'

X_PATH = 'data/'

X_LABELS = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'p', 'y', 'r']

Y_FILE = 'y_train.csv'

SUBJECT_FILE = 'subject.csv'

LOG_DIR= ''
