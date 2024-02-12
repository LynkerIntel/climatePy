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

def test_gpm_imerg_hdf5(catalog_data):
    
    bb = make_aoi("fort_collins")
    
    URL = 'https://gpm1.gesdisc.eosdis.nasa.gov/opendap/hyrax/GPM_L3/GPM_3IMERGHHL.06/2024/018/3B-HHR-L.MS.MRG.3IMERG.20240118-S000000-E002959.0000.V06E.HDF5'
    
    # TODO: Need to make fix in read_dap_file(), dap_xyz(), _resource_time() functions to deal with this situation
    # TODO: Has to do with ordering of dimensions in the HDF5 file, climateR had the same issue and resolved it.
    with pytest.raises(Exception):
        plp = climatePy._dap.dap(URL = URL, varname = 'probabilityLiquidPrecipitation', AOI = bb, verbose=True)


def test_gridmet_dap(catalog_data):

    # Get the bounding box for Fort Collins, CO as a geopandas dataframe
    bb = make_aoi("fort_collins")
    # catalog_data = catalog_data()
    # catalog_data.id.unique()[1500]
    cat = climatePy._climatepy_filter.climatepy_filter(id = 'cfsv2_gridmet', AOI = bb)
  
    one_cat = cat.iloc[0:1]

    burnindex_g1 = climatePy._dap.dap(catalog = one_cat, AOI = bb, verbose = True)
 
    assert len(burnindex_g1.keys()) == 1
    assert type(burnindex_g1) != list
    assert type(burnindex_g1["bi"]) == xr.DataArray
    assert list(burnindex_g1.keys()) == ["bi"]

    assert burnindex_g1["bi"].dims == ("y", "x", "time")
    assert len(burnindex_g1["bi"].x) == 5
    assert len(burnindex_g1["bi"].y) == 5
    assert len(burnindex_g1["bi"].time) == 28


    three_cat = cat.iloc[0:3]
    three_daps = climatePy._dap.dap(catalog = three_cat, AOI = bb, verbose = True)

    assert len(three_daps.keys()) == 3
    assert type(three_daps) == dict
    assert type(three_daps["bi"]) == xr.DataArray
    assert list(three_daps.keys()) == ['bi', 'erc', 'fm100']

    assert three_daps["bi"].dims == ("y", "x", "time")
    assert len(three_daps["bi"].x) == 5
    assert len(three_daps["bi"].y) == 5
    assert len(three_daps["bi"].time) == 28

    assert three_daps["erc"].dims == ("y", "x", "time")
    assert len(three_daps["erc"].x) == 5
    assert len(three_daps["erc"].y) == 5
    assert len(three_daps["erc"].time) == 28

# def test_chirps_daily(catalog_data):
#     # ---- Test 1: Test chirps20GlobalDailyP05 ----
#     # ---- Fort Collins, CO ----

#   # Get the bounding box for Fort Collins, CO as a geopandas dataframe
#     bb = make_aoi("fort_collins")

#     # catalog_data = catalog_data()
#     # tmp = catalog_data[catalog_data['id'] == "chirps20GlobalAnnualP05"].iloc[0:1]

#     cat = climatePy._climatepy_filter.climatepy_filter(id = 'chirps20GlobalDailyP05')

#     # NOTE: CHIRPS data is currently inaccesible due to server issues with the data provider
#     # NOTE: This test will fail until the data provider resolves the server issues
#     with pytest.raises(Exception):
#         chirps_daily = climatePy._dap.dap(catalog = cat, 
#                                         AOI = bb,  
#                                         startDate = "2011-11-29",
#                                         endDate = "2011-12-03")
#     # assert len(chirps_daily) == 1
#     # assert type(chirps_daily) != list
#     # assert chirps_daily.keys() == "precip"
#     # assert type(chirps_daily["precip"]) == xr.DataArray

#     # assert chirps_daily["precip"].dims == ("y", "x", "time")
#     # assert len(chirps_daily["precip"].x) == 5
#     # assert len(chirps_daily["precip"].y) == 5
#     # assert len(chirps_daily["precip"].time) == 5
#     # assert chirps_daily["precip"].time.values.tolist() == ['precip_2011-11-29-00-00-00', 'precip_2011-11-30-00-00-00', 'precip_2011-12-01-00-00-00', 'precip_2011-12-02-00-00-00', 'precip_2011-12-03-00-00-00']

# def test_chirps_pentad(catalog_data):
#     # ---- Test 1: Test chirps20GlobalPentadP05 ----
#     # ---- Fort Collins, CO ----

#   # Get the bounding box for Fort Collins, CO as a geopandas dataframe
#     bb = make_aoi("fort_collins")

#     # catalog_data = catalog_data()
#     # tmp = catalog_data[catalog_data['id'] == "chirps20GlobalAnnualP05"].iloc[0:1]

#     cat = climatePy._climatepy_filter.climatepy_filter(id = 'chirps20GlobalPentadP05')

#     # NOTE: CHIRPS data is currently inaccesible due to server issues with the data provider
#     # NOTE: This test will fail until the data provider resolves the server issues
#     with pytest.raises(Exception):
#        chirps_pentad = climatePy._dap.dap(catalog = cat, 
#                                           AOI = bb,  
#                                           startDate = "2011-11-29",
#                                           endDate = "2012-05-03")
       
# def test_chirps_monthly(catalog_data):

#     # ---- Test 1: Test chirps20GlobalMonthlyP05 ----
#     # ---- Fort Collins, CO ----

#   # Get the bounding box for Fort Collins, CO as a geopandas dataframe
#     bb = make_aoi("fort_collins")

#     # catalog_data = catalog_data()
#     # tmp = catalog_data[catalog_data['id'] == "chirps20GlobalAnnualP05"].iloc[0:1]

#     cat = climatePy._climatepy_filter.climatepy_filter(id = 'chirps20GlobalMonthlyP05')

#     # NOTE: CHIRPS data is currently inaccesible due to server issues with the data provider
#     # NOTE: This test will fail until the data provider resolves the server issues
#     with pytest.raises(Exception):
#        chirps_monthly = climatePy._dap.dap(catalog = cat, 
#                                           AOI = bb,  
#                                           startDate = "2011-11-01",
#                                           endDate = "2012-05-01")
       
# def test_chirps_annual(catalog_data):

#     # ---- Test 1: Test chirps20GlobalAnnualP05 ----
#     # ---- Fort Collins, CO ----

#   # Get the bounding box for Fort Collins, CO as a geopandas dataframe
#     bb = make_aoi("fort_collins")

#     # catalog_data = catalog_data()
#     # tmp = catalog_data[catalog_data['id'] == "chirps20GlobalAnnualP05"].iloc[0:1]

#     cat = climatePy._climatepy_filter.climatepy_filter(id = 'chirps20GlobalAnnualP05')

#     # NOTE: CHIRPS data is currently inaccesible due to server issues with the data provider
#     # NOTE: This test will fail until the data provider resolves the server issues
#     with pytest.raises(Exception):
#         chirps_yearly = climatePy._dap.dap(catalog = cat, 
#                                             AOI = bb,  
#                                             startDate = "2011-01-01",
#                                             endDate = "2013-01-01")
    
#     # assert len(chirps_yearly) == 1
#     # assert type(chirps_yearly) != list
#     # assert list(chirps_yearly.keys())[0] == "precip"
#     # assert type(chirps_yearly["precip"]) == xr.DataArray
#     # assert len(chirps_yearly["precip"].x) == 5
#     # assert len(chirps_yearly["precip"].y) == 5
#     # assert chirps_yearly["precip"].time.values.tolist() == ['precip_2010-12-31-00-00-00', 'precip_2011-12-31-12-00-00', 'precip_2012-12-31-00-00-00']
    
#     # ---- Test 2: Test chirps20GlobalAnnualP05 ----
#     # ---- Florida, USA ----

#     # Florida bounding box
#     florida = make_aoi("florida")

#     # NOTE: CHIRPS data is currently inaccesible due to server issues with the data provider
#     # NOTE: This test will fail until the data provider resolves the server issues
#     with pytest.raises(Exception):
#         chirps_yearly_florida = climatePy._dap.dap(catalog = cat, 
#                                             AOI = florida,  
#                                             startDate = "2011-01-01",
#                                             endDate = "2013-01-01")
    
#     # assert len(chirps_yearly_florida) == 1
#     # assert type(chirps_yearly_florida) != list
#     # assert type(chirps_yearly_florida) == dict
#     # assert list(chirps_yearly_florida.keys())[0] == "precip"
#     # assert type(chirps_yearly_florida["precip"]) == xr.DataArray

#     # assert chirps_yearly_florida["precip"].time.values.tolist() == ['precip_2010-12-31-00-00-00', 'precip_2011-12-31-12-00-00', 'precip_2012-12-31-00-00-00']
#     # assert chirps_yearly_florida["precip"].attrs["varname"] == "precip"
#     # assert chirps_yearly_florida["precip"].attrs["variable"] == "precip"
