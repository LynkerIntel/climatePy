# __init__.py
import pandas as pd
import pkg_resources

# warnings lib
import warnings

# suppress warnings
warnings.filterwarnings('ignore', category=Warning)

def params():
    data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.csv')
    # data_file = pkg_resources.resource_filename('src', 'data/catalog.csv')
    data = pd.read_csv(data_file, low_memory=False)
    return data

from ._climatepy_filter import climatepy_filter
from ._dap import dap, dap_crop, dap_get
from ._shortcuts import getTerraClim, getTerraClimNormals, getGridMET, getMACA, \
    get3DEP, getLOCA, getPRISM, getPolaris, \
    getBCCA, getLivneh, getLivneh_fluxes, getISRIC_soils, getDaymet, \
    getVIC, getNASADEM, getWorldClim, getCHIRPS, getLCMAP, getNLDAS

# can i do this and get the __all__ to grab all the functions from ._shortcuts.py ?
# from ._shortcuts import *

__all__ = [
    'climatepy_filter',
    'dap',
    'dap_crop',
    'dap_get',
    'getTerraClim',
    'getTerraClimNormals', 
    'getGridMET', 
    'getMACA', 
    'get3DEP', 
    'getLOCA', 
    'getPRISM',
    'getPolaris', 
    'getBCCA',
    'getLivneh', 
    'getLivneh_fluxes', 
    'getISRIC_soils', 
    'getDaymet',
    'getVIC', 
    'getNASADEM', 
    'getWorldClim', 
    'getCHIRPS', 
    'getLCMAP',
    'getNLDAS',
    'params'
]

##############################
# Old method
# import pandas as pd
# import pkg_resources

# def params():
#     data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.csv')
#     # data_file = pkg_resources.resource_filename('src', 'data/catalog.csv')
#     data = pd.read_csv(data_file)
#     return data

# from ._climatepy_filter import *
# from ._dap import *
# from ._netrc_utils import *
# from ._shortcuts import *
# from ._utils import *