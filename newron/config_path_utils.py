import os
from pathlib import Path


def find_folder_in_folder(name, path):
    for dirs in os.listdir(path):
        # Do not use os.walk here, because we only want to search in the current folder
        if dirs == name:
            return os.path.join(path, dirs)


def find_file(name, path):
    # Finds file in the current directory or their subdirectories
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def find_config_folder(config_folder_name=".newron", path=None):
    # Finds config folder in the current directory or in the parent directory or their parent directory till root,
    # does not search in subdirectories
    if path is None:
        path = Path(__file__).parent

    if path == path.parent:
        return None

    config_dir = find_folder_in_folder(config_folder_name, path)

    if config_dir:
        return config_dir
    else:
        path = path.parent
        return find_config_folder(config_folder_name, path)


def find_config_file(config_file_name, path=None):
    # Finds config file in the current directory or in the parent directory or their parent directory till root,
    config_folder = find_config_folder(path=path)
    if config_folder:
        return find_file(config_file_name, config_folder)
    else:
        return None


def get_path_in_home_dir(file_name="", path=".newron"):
    return os.path.join(os.path.expanduser("~"), path, file_name)
