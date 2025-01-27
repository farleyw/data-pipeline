""" This module stores global variable information that are needed to run upt
variables. Update the values contained here and store this file in the same 
folder as your notebooks. The notebooks will call this stored information with
each run.
"""

import os
import json

calibration_ratio = 2.7488  # pixels/um (feature extraction v4 value)
# or 3.4 if using the feature extraction v2

config_info = {"blob_storage_name": None,
               "connection_string": None,
               "server": None,
               "database": None,
               "db_user": None,
               "db_password": None,
               'subscription_id': None,
               'resource_group': None,
               'workspace_name': None,
               'experiment_name': None,
               'api_key': None,
               'model_name': None,
               'endpoint_name': None,
               'deployment_name': None,
               }

default_investigators = {"Firstname_Lastname": ['Organization',
                            'email@org.com'],
                         "Firstname_Lastname": ['Organization', 
                            'email@org.com'],
                         }