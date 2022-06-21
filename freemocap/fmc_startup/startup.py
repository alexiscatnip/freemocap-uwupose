from pathlib import Path

from ruamel.yaml import YAML

from freemocap import recordingconfig
from freemocap.fmc_startup import startupGUI

import tkinter as tk
from tkinter import filedialog


def get_user_preferences(session):
    """
    load user preferences if they exist, create a new preferences yaml if they don't
    """
    path_to_this_py_file = Path(__file__).parent
    preferences_path = path_to_this_py_file/'user_preferences.yaml' 
    preferences_yaml = YAML()

    # check for a user preferences yaml, if it doesn't exist,
    # build one using the default parameters in recordingconfig.py
    if preferences_path.exists():
        preferences = preferences_yaml.load(preferences_path)
    else:
        preferences = recordingconfig.parameters_for_yaml
        preferences_yaml.dump(preferences, preferences_path)
    
    session.preferences = preferences
    session.preferences_path = preferences_path
