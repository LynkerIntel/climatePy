# __init__.py
import pandas as pd
import pyarrow

import pkg_resources

# warnings lib
import warnings

# suppress warnings
warnings.filterwarnings('ignore', category=Warning)

def data_catalog():
    
    data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.parquet')
    data = pd.read_parquet(data_file)
    # data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.csv')
    # data = pd.read_csv(data_file, low_memory=False)

    return data

from ._climatepy_filter import climatepy_filter
from ._dap import dap, dap_crop, dap_get, vrt_crop_get, crop_vrt
from ._shortcuts import getTerraClim, getTerraClimNormals, getGridMET, getMACA, \
    get3DEP, getLOCA, getPRISM, getPolaris, \
    getBCCA, getLivneh, getLivneh_fluxes, getISRIC_soils, getDaymet, \
    getVIC, getNASADEM, getWorldClim, getCHIRPS, getLCMAP, getNLDAS, getGLDAS, getMODIS
from ._netrc_utils import writeDodsrc, writeNetrc, getNetrcPath, getDodsrcPath, checkNetrc, checkDodsrc
from ._extract_sites import extract_sites
from ._viz import animiation_raster

__all__ = [
    'data_catalog',
    'climatepy_filter',
    'dap',
    'dap_crop',
    'dap_get',
    'vrt_crop_get', 
    'crop_vrt', 
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
    'getGLDAS',
    'getMODIS',
    'extract_sites',
    'animiation_raster',
    'writeDodsrc',
    'writeNetrc',
    'getNetrcPath',
    'getDodsrcPath',
    'checkNetrc',
    'checkDodsrc'
]

##############################

# # # try and get up to date catalog from GitHub, otherwise use local catalog file
# def data_catalog():
#     url = 'https://github.com/mikejohnson51/climateR-catalogs/releases/latest/download/catalog.parquet'
#     cat = None
#     try:
#         cat = pd.read_parquet(url)
#     except Exception:
#         url = pkg_resources.resource_filename('climatePy', 'data/catalog.parquet')
#         cat = pd.read_parquet(url)
#     return cat

# # save new catalog to local file
# def save_new_data_catalog():
#     url = 'https://github.com/mikejohnson51/climateR-catalogs/releases/latest/download/catalog.parquet'
#     cat = None
#     cat = pd.read_parquet(url)
#     cat.to_parquet(pkg_resources.resource_filename('climatePy', 'data/catalog.parquet'))

##############################
##############################
# # Old method
# import pandas as pd
# import pkg_resources

# def data_catalog():
#     data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.csv')
#     # data_file = pkg_resources.resource_filename('src', 'data/catalog.csv')
#     data = pd.read_csv(data_file)
#     return data

# from ._climatepy_filter import *
# from ._dap import *
# from ._netrc_utils import *
# from ._shortcuts import *
# from ._utils import *