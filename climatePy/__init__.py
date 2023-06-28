# __init__.py
import pandas as pd
import pkg_resources

def params():
    data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.csv')
    # data_file = pkg_resources.resource_filename('src', 'data/catalog.csv')
    data = pd.read_csv(data_file)
    return data

from .climatepy_filter import *
from .dap import *
from .netrc_utils import *
from .shortcuts import *
from .utils import *