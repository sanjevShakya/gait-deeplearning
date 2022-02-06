from pathlib import Path
from venv import create
from gait.constants import ROOT_DATA_DIR, LOGS_DIR, MODELS_DIR
from gait.utils import create_dir

def get_project_root():
    return Path(__file__).parent.parent


def get_data_folder():
    return get_project_root().joinpath(ROOT_DATA_DIR)


def get_log_folder():
    return get_project_root().joinpath(LOGS_DIR)


def get_model_folder():
    return get_project_root().joinpath(MODELS_DIR)


def get_log_file_path(overlap_percent, filename):
    dir_path = get_log_folder().joinpath('log_{}_overlap'.format(overlap_percent))
    file_path = dir_path.joinpath(filename)
    return (dir_path, file_path)

def get_model_file_path(overlap_percent, filename):
    dir_path = get_model_folder().joinpath('model_{}_overlap'.format(overlap_percent))
    create_dir(dir_path)
    file_path = dir_path.joinpath(filename);
    return file_path

    