# pytest library
import pytest

# # import climatePy modules
# import climatePy

# data manipulation libs
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

# standard python libs
import re
from datetime import datetime


# old imports
# import climatePy._utils as climatePy
# import climatePy._extract_sites as climatePy
# import climatePy._dap as climatePy
# import climatePy._shortcuts as climatePy


import climatePy._climatepy_filter
# import climatePy._climatepy_filter as climatePy

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

# Helper function to make AOI from hardcoded bounding box coordinates
# provide a "name" to the function to return a bounding box geopandas dataframe
# name options: "north_carolina", "kenya", "harvey", "michigan", "colorado", "florida", "egypt"
def make_aoi(name):

  bb_map = {"north_carolina": [-84.32182, 33.75288, -75.40012,  36.58814 ],
  "kenya": [33.89357, -4.67677, 41.85508,  5.50600 ],
  "harvey": [-106.64548,   24.39631,  -79.97431,   36.50070 ],
  "michigan": [-90.41839,  41.69612, -82.12297,  48.30606 ],
  "colorado": [-109.06019,  36.99246, -102.04152,  41.00342 ],
  "florida": [-87.63479,  24.39631, -79.97431,  31.00097 ],
  "egypt": [24.70007, 22.00000, 36.86623, 31.58568 ]
  }

  if name not in bb_map:
    raise Exception("AOI not found")

  xmin = bb_map[name][0]
  ymin = bb_map[name][1]
  xmax = bb_map[name][2]
  ymax = bb_map[name][3]

  # make bounding box
  # AOI = box(xmin, ymin, xmax, ymax)

  aoi = gpd.GeoDataFrame({
    "geometry" : [box(xmin, ymin, xmax, ymax)]
    }, 
    crs = "EPSG:4326"
    )
    
  return aoi

def test_climatepy_filter(AOI):

    # correctly return a single asset
    assert len(climatePy._climatepy_filter.climatepy_filter(asset='2019 Land Cover L48')) == 1

    # raise an error, id not in catalog
    with pytest.raises(Exception):
        climatePy._climatepy_filter.climatepy_filter(id="BLAH")

    # raise an error, asset not in catalog
    with pytest.raises(Exception):
        climatePy._climatepy_filter.climatepy_filter(id="MODIS", asset="BLAH")
    assert len(climatePy._climatepy_filter.climatepy_filter(id="maca_day", varname="pr", model=1, startDate="2000-01-01")) == 1
    
    # raise an error, model not in catalog
    with pytest.raises(Exception):
        climatePy._climatepy_filter.climatepy_filter(id="maca_day", varname="pr", model=1e9, startDate="2000-01-01")
    
    # raise an error, landcover not in US
    with pytest.raises(Exception):
        climatePy._climatepy_filter.climatepy_filter(AOI=make_aoi("egypt"), asset='2019 Land Cover L48')

    # correctly return a single asset
    assert len(climatePy._climatepy_filter.climatepy_filter(id="gridmet", varname="pr")) == 1

    # correctly return multiple assets
    assert len(climatePy._climatepy_filter.climatepy_filter(id="bcca", varname=['pr', 'tasmax', 'tasmin'],
                                ensemble='r1i1p1', model=["CSIRO-Mk3-6-0"],
                                scenario=['rcp45', 'rcp85'], startDate="2079-10-01")) == 6
    # raise an error, ensemble not in catalog
    with pytest.raises(Exception):
        climatePy._climatepy_filter.climatepy_filter(id="bcca", varname=['pr', 'tasmax', 'tasmin'],
                        model=["CSIRO-Mk3-6-0"], ensemble="XXXX",
                        scenario=['rcp45', 'rcp85'], startDate="2079-10-01")

def test_ensemble():
    # NULL
    x = climatePy._climatepy_filter.climatepy_filter(id="loca", varname="tasmin", model='GISS-E2-R', scenario='rcp45', startDate="2050-01-01")
    assert len(x) == 1

    # Declared Wrong
    with pytest.raises(Exception):
        climatePy._climatepy_filter.climatepy_filter(id="loca", varname="tasmin", model='GISS-E2-R', scenario='rcp45', ensemble='r1i1p1', startDate="2050-01-01")

    # Declared
    x = climatePy._climatepy_filter.climatepy_filter(id="loca", varname="tasmin", model='GISS-E2-R', scenario='rcp45', ensemble='r6i1p1', startDate="2050-01-01")
    assert len(x) == 1

    # Declared, Multi-model, one return
    x = climatePy._climatepy_filter.climatepy_filter(id="loca", varname="tasmin", model=['GISS-E2-R', 'ACCESS1-0'], scenario='rcp45', ensemble='r6i1p1', startDate="2050-01-01")
    assert len(x) == 1

    # Declared, Multi-model, 2 return
    x = climatePy._climatepy_filter.climatepy_filter(id="loca", varname="tasmin", model=['ACCESS1-3', 'ACCESS1-0'], scenario='rcp45', ensemble='r1i1p1', startDate="2050-01-01")
    assert len(x) == 2

    # NULL, Multi model, 2 return
    x = climatePy._climatepy_filter.climatepy_filter(id="loca", varname="tasmin", model=['GISS-E2-R', 'ACCESS1-0'], scenario='rcp45', startDate="2050-01-01")
    assert len(x) == 2
