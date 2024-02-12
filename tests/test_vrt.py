# pytest library
import pytest

# data manipulation libs
import pandas as pd
import geopandas as gpd
import xarray as xr
from shapely.geometry import box

# standard python libs
import re
from datetime import datetime

# libraries for importing climate data catalog
import pyarrow
import pkg_resources

# # # import climatePy modules
# import climatePy._dap as climatePy
# import climatePy._climatepy_filter as climatePy

import climatePy._dap 
import climatePy._climatepy_filter

# # import climatePy library from downloaded package
# import climatePy

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

@pytest.fixture
def catalog_data():
    
    data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.parquet')
    data = pd.read_parquet(data_file)
    # data_file = pkg_resources.resource_filename('climatePy', 'data/catalog.csv')
    # data = pd.read_csv(data_file, low_memory=False)

    return data

# Helper function to make AOI from hardcoded bounding box coordinates
# provide a "name" to the function to return a bounding box geopandas dataframe
# name options: "north_carolina", "kenya", "harvey", "michigan", "colorado", "florida", "egypt"
def make_aoi(name):
  
  # bb_map key/value pairs: 
    # key = place name (str, e.g. "north_carolina")
    # value = [xmin, ymin, xmax, ymax]
  bb_map = {"north_carolina": [-84.32182, 33.75288, -75.40012,  36.58814 ],
  "kenya": [33.89357, -4.67677, 41.85508,  5.50600 ],
  "harvey": [-106.64548,   24.39631,  -79.97431,   36.50070 ],
  "michigan": [-90.41839,  41.69612, -82.12297,  48.30606 ],
  "colorado": [-109.06019,  36.99246, -102.04152,  41.00342 ],
  "florida": [-87.63479,  24.39631, -79.97431,  31.00097 ],
  "egypt": [24.70007, 22.00000, 36.86623, 31.58568 ],
  "fort_collins": [-105.16062, 40.49747, -104.99262, 40.66547]
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

def test_vrt(catalog_data):

    #         URL (str or list, optional): The URL(s) of the VRT file(s) to open. If not provided,
    #  it is extracted from the catalog.
    #         catalog (object, optional): The catalog object containing the URL(s) of the VRT file(s). Required if URL is not provided.
    #         AOI (geopandas.GeoDataFrame, optional): The Area of Interest polygon to crop the VRT data to.
    #         grid (object, optional): The grid object defining the extent and CRS for cropping and reprojection.
    #         varname (str, optional): The name of the variable to select from the VRT data.
    #         start (int, optional): The start index for subsetting bands in the VRT data.
    #         end (int, optional): The end index for subsetting bands in the VRT data.
    #         toptobottom (bool, optional): Whether to flip the data vertically.
    #         verbose (bool, optional): Whether to print informative messages during processing. Default is False
    
    # Get the bounding box for Fort Collins, CO as a geopandas dataframe
    bb = make_aoi("fort_collins")

    # catalog_data = catalog_data()
    tmp = catalog_data[catalog_data['id'] == "HBV"].iloc[0:1]
    # catalog_data[catalog_data['id'] == "HBV"].varname.unique()
    # type(tmp["varname"].values)

    # x = "vagsdgd"
    # import numpy as np
    # # convert x to a numpy.ndarray as an object
    # np.array([x], dtype=object)

    hbv = climatePy._dap.vrt_crop_get(catalog=tmp, AOI=bb)
    
    hbv1 = climatePy._dap.vrt_crop_get(catalog=tmp, AOI=bb, start=2)
    hbv1_2 = climatePy._dap.vrt_crop_get(catalog=tmp, AOI=bb, varname="FC")

    hbv_2 = climatePy._dap.vrt_crop_get(catalog=tmp, AOI=bb, start=2, end=3)

    assert list(hbv.keys()) == ["HBV"]
    assert len(hbv) == 1
    assert type(hbv["HBV"]) == dict
    assert type(hbv["HBV"]["beta"]) == xr.DataArray
    assert list(hbv["HBV"]) == ['beta', 'FC', 'K0', 'K1', 'K2', 'LP', 'PERC', 'UZL', 'TT', 'CFMAX', 'CFR', 'CWH']

    assert hbv.keys() == hbv1.keys()
    assert hbv.keys() == hbv1_2.keys()
    assert hbv.keys() == hbv_2.keys()

    assert hbv1.keys() == hbv1_2.keys()
    assert hbv1.keys() == hbv_2.keys()

    assert hbv_2.keys() == hbv.keys()

    assert hbv_2["HBV"]["beta"].equals(hbv1_2["HBV"]["beta"])
    assert hbv1["HBV"]["K0"].equals(hbv1_2["HBV"]["K0"])

def test_nlcd_vrt(catalog_data):

    # Get the bounding box for Fort Collins, CO as a geopandas dataframe
    bb = make_aoi("fort_collins")

    cat = climatePy._climatepy_filter.climatepy_filter(asset = '2019 Land Cover L48')

    nlcd = climatePy._dap.dap(catalog = cat, AOI = bb)

    assert len(nlcd) == 1
    assert type(nlcd) != list
    assert type(nlcd["Land_Cover"]) == xr.DataArray
    assert list(nlcd.keys()) == ["Land_Cover"]