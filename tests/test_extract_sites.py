import pytest

import pandas as pd
import xarray as xr
import geopandas as gpd
import numpy as np
import re

# import climatePy modules
import climatePy._extract_sites as climatePy
# import climatePy

# make some testing data
@pytest.fixture
def df():
    return pd.DataFrame({'time': ['pr_2018-01-01-00-00-00', 'pr_2018-01-02-00-00-00', 'pr_2018-01-03-00-00-00']})

@pytest.fixture
def df_expected():
    return pd.DataFrame({'date': ['2018-01-01T00:00:00', '2018-01-02T00:00:00', '2018-01-03T00:00:00'],'varname': ['pr', 'pr', 'pr']})

@pytest.fixture
def col():
    return 'time'

@pytest.fixture
def clean_time_expected():
    return [np.datetime64('2018-01-01T00:00:00'), np.datetime64('2018-01-02T00:00:00'), np.datetime64('2018-01-03T00:00:00')]

@pytest.fixture
def pts():
    gdf = gpd.GeoDataFrame(geometry=gpd.points_from_xy([0, 1, 2], [0, 1, 2]))
    gdf.crs = "EPSG:4326"
    return gdf

@pytest.fixture
def r():
    return xr.DataArray(np.random.rand(3, 3), dims=('time', 'x'), coords={'time': pd.date_range('2018-01-01', periods=3), 'x': [0, 1, 2]})

# currently poorly written unit tests
def test_clean_time(df, df_expected, clean_time_expected):
    df['date'] = climatePy.clean_time(df, 'time', inplace= False)
    assert df['date'].tolist() == df_expected['date'].tolist()
    cleaned_time = climatePy.clean_time(df, "time", inplace=False)
    assert cleaned_time == clean_time_expected

def test_clean_varname(df, col, df_expected):
    cleaned_varname_df = climatePy.clean_varname(df, col)
    assert cleaned_varname_df['varname'].equals(df_expected['varname'])
    cleaned_varname = climatePy.clean_varname(df, col, inplace=False)
    assert np.array_equal(cleaned_varname, df_expected['varname'].values)

def test_pts_extracter(r, pts, df_expected):
    extracted_df = climatePy.pts_extracter(r, pts)
    assert extracted_df.equals(df_expected)

def test_extract_sites_single_dataarray(r, pts, df_expected):
    extracted_df = climatePy.extract_sites(r, pts)
    assert extracted_df.equals(df_expected)

def test_extract_sites_dict(r, pts, df_expected):
    data_dict = {'var1': r, 'var2': r}
    expected_df = pd.concat([df_expected, df_expected], ignore_index=True)
    extracted_df = climatePy.extract_sites(data_dict, pts)
    assert extracted_df.equals(expected_df)
