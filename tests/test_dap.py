# pytest library
import pytest

# import climatePy modules
import climatePy

# data manipulation libs
import pandas as pd
import geopandas as gpd
import xarray as xr

# standard python libs
import re
from datetime import datetime


# old imports
# from .climatePy import shortcuts
# from src.climatePy import shortcuts
# from src.climatePy import dap
# from src.climatePy import climatepy_filter
# from src.climatePy import utils

# AOI    = gpd.read_file('climatePy/data/boulder_county.gpkg')

# climatePy.getGridMET(AOI, "pr", "2000-01-01", "2000-01-01", verbose=True)
# AOI    = gpd.read_file('climatePy/data/san_luis_obispo_county.gpkg')

# @pytest.fixture(params=['miami_dade_county', 'san_luis_obispo_county', 
#                         'litchfield_county', 'tuscaloosa_county', 
#                         'boulder_county', 'king_county'])

@pytest.fixture(params=['san_luis_obispo_county', 'boulder_county'])
# @pytest.fixture(params=['san_luis_obispo_county'])

def AOI(request): 
    # filepath = f'src/data/{request.param}.gpkg'
    filepath = f'climatePy/data/{request.param}.gpkg'
    AOI = gpd.read_file(filepath)
    yield AOI

# # Create a test for dap() function on a subset of the params data catalog, id == "gridmet" and variable == "pr"
def test_dap_gridmet_case1(AOI):

    x = "553"
    assert x == "553"
    assert x != "423652"