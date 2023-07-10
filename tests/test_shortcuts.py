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

def AOI(request): 
    # filepath = f'src/data/{request.param}.gpkg'
    filepath = f'climatePy/data/{request.param}.gpkg'
    AOI = gpd.read_file(filepath)
    yield AOI

def test_getTerraClim_case1(AOI):

    varname = ["ppt", "tmax"]
    startDate = "2018-01-01"
    endDate   = "2018-03-02"
    verbose   = True

    # Call function to get output
    output = climatePy.getTerraClim( AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"ppt", "tmax"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["ppt"], xr.DataArray)
    assert isinstance(output["tmax"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["ppt"].dims == ("y", "x", "time")
    assert output["tmax"].dims == ("y", "x", "time")   

    dates1 = [datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values]
    dates2 = [datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values]
    
    start_date1 = min(dates1)
    end_date1   = max(dates1)
    start_date2 = min(dates2)
    end_date2   = max(dates2)

    assert end_date1 - start_date1 == pd.Timedelta("59D")
    assert end_date2 - start_date2 == pd.Timedelta("59D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")

    assert "ppt" in output["ppt"].time[0].values.item()
    assert "tmax" in output["tmax"].time[0].values.item()

def test_getGridMET_case1(AOI):
    # ---- Case 1 ----
    verbose   = True
    varname = "pr"
    startDate="2000-01-01"
    endDate="2000-01-01"
    
    # Call function to get output
    output = climatePy.getGridMET( AOI, varname, startDate, endDate, verbose)
    
    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"pr"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["pr"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["pr"].time.values])
    
    assert end_date1 - start_date1 == pd.Timedelta("0D")

    assert "pr" in output["pr"].time[0].values.item()
    assert len(output["pr"].time.values) == 1

def test_getGridMET_case2(AOI):    
    # ---- Case 2 ----
    verbose   = True
    varname = ["pr", "tmmx"]
    startDate = "2018-01-01"
    endDate   = "2018-01-02"

    # Call function to get output
    output = climatePy.getGridMET( AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"tmmx", "pr"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)
    assert isinstance(output["tmmx"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")
    assert output["tmmx"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["pr"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["pr"].time.values])
    start_date2 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmmx"].time.values])
    end_date2   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmmx"].time.values])
    
    assert end_date1 - start_date1 == pd.Timedelta("1D")
    assert end_date2 - start_date2 == pd.Timedelta("1D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")

    assert "pr" in output["pr"].time[0].values.item()
    assert "tmmx" in output["tmmx"].time[0].values.item()
    assert len(output["pr"].time.values) == 2
    assert len(output["tmmx"].time.values) == 2

def test_getGridMET_case3(AOI):
    # ---- Case 3 ----
    verbose = True
    varname = ["pr", "tmmx", "rmin"]
    startDate = "2015-01-01"
    endDate   = "2016-02-05"

    # Call function to get output
    output = climatePy.getGridMET( AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"tmmx", "pr", "rmin"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)
    assert isinstance(output["tmmx"], xr.DataArray)
    assert isinstance(output["rmin"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")
    assert output["tmmx"].dims == ("y", "x", "time")
    assert output["rmin"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["pr"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["pr"].time.values])
    start_date2 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmmx"].time.values])
    end_date2   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmmx"].time.values])
    start_date3 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["rmin"].time.values])
    end_date3   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["rmin"].time.values])

    assert end_date1 - start_date1 == pd.Timedelta("400D")
    assert end_date2 - start_date2 == pd.Timedelta("400D")
    assert end_date3 - start_date3 == pd.Timedelta("400D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")
    assert end_date3 - end_date2 == pd.Timedelta("0D")

    assert "pr" in output["pr"].time[0].values.item()
    assert "tmmx" in output["tmmx"].time[0].values.item()
    assert "rmin" in output["rmin"].time[0].values.item()

    assert len(output["pr"].time.values) == 401
    assert len(output["tmmx"].time.values) == 401
    assert len(output["rmin"].time.values) == 401
    
def test_getGridMET_case4(AOI):
    # ---- Case 4 ----
    verbose = True
    varname = ["rmin"]
    startDate = "2005-01-01"
    endDate   = "2010-01-01"

    # Call function to get output
    output = climatePy.getGridMET(AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"rmin"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["rmin"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["rmin"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["rmin"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["rmin"].time.values])

    assert end_date1 - start_date1 == pd.Timedelta("1826D")

    assert "rmin" in output["rmin"].time[0].values.item()
    assert len(output["rmin"].time.values) == 1827

    
def test_getTerraClimNormals_case1(AOI):
    
    # ---- Case 1 ----
    verbose   = True
    varname = ["ppt", "tmax"]
    scenario  = '19611990'
    month     = [1, 2]

    # call function
    output = climatePy.getTerraClimNormals( AOI, varname, month, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"ppt", "tmax"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["ppt"], xr.DataArray)
    assert isinstance(output["tmax"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["ppt"].dims == ("y", "x", "time")
    assert output["tmax"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    start_date2 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    end_date2   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    
    assert end_date1 - start_date1 == pd.Timedelta("31D")
    assert end_date2 - start_date2 == pd.Timedelta("31D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")

    # check time variables names are correct
    assert "ppt" in output["ppt"].time[0].values.item()
    assert "tmax" in output["tmax"].time[0].values.item()
    assert scenario in output["ppt"].time[0].values.item()
    assert scenario in output["tmax"].time[0].values.item()
    assert len(output["ppt"].time.values) == 2
    assert len(output["tmax"].time.values) == 2

def test_getTerraClimNormals_case2(AOI):

    # ---- Case 2 ----
    verbose   = True
    varname = ["ppt", "tmax"]
    scenario  = '19812010'
    month     = [i for i in range(1, 13)]

    # call function
    output = climatePy.getTerraClimNormals( AOI, varname, month, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"ppt", "tmax"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["ppt"], xr.DataArray)
    assert isinstance(output["tmax"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["ppt"].dims == ("y", "x", "time")
    assert output["tmax"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    start_date2 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    end_date2   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    
    assert end_date1 - start_date1 == pd.Timedelta("334D")
    assert end_date2 - start_date2 == pd.Timedelta("334D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")

    # check time variables names are correct
    assert "ppt" in output["ppt"].time[0].values.item()
    assert "tmax" in output["tmax"].time[0].values.item()
    assert scenario in output["ppt"].time[0].values.item()
    assert scenario in output["tmax"].time[0].values.item()
    assert len(output["ppt"].time.values) == 12
    assert len(output["tmax"].time.values) == 12

def test_getTerraClimNormals_case3(AOI):
    # ---- Case 3 ----
    verbose   = True
    varname = ["ppt", "tmax"]
    scenario = '2C'
    month     = [1, 3]

    # call function
    output = climatePy.getTerraClimNormals( AOI, varname, month, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"ppt", "tmax"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["ppt"], xr.DataArray)
    assert isinstance(output["tmax"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["ppt"].dims == ("y", "x", "time")
    assert output["tmax"].dims == ("y", "x", "time")

    # Assert dates are correct
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    start_date2 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    end_date2   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    
    assert end_date1 - start_date1 == pd.Timedelta("59D")
    assert end_date2 - start_date2 == pd.Timedelta("59D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")

    # check time variables names are correct
    assert "ppt" in output["ppt"].time[0].values.item()
    assert "tmax" in output["tmax"].time[0].values.item()
    assert scenario in output["ppt"].time[0].values.item()
    assert scenario in output["tmax"].time[0].values.item()
    assert len(output["ppt"].time.values) == 3
    assert len(output["tmax"].time.values) == 3
    
def test_getTerraClimNormals_case4(AOI):
    # ---- Case 4 ----
    verbose   = True
    varname = ["ppt", "tmax"]
    scenario = '4C'
    month     = [1]
    output = climatePy.getTerraClimNormals( AOI, varname, month, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"ppt", "tmax"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["ppt"], xr.DataArray)
    assert isinstance(output["tmax"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["ppt"].dims == ("y", "x", "time")
    assert output["tmax"].dims == ("y", "x", "time")   

    # check dates
    start_date1 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    end_date1   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["ppt"].time.values])
    start_date2 = min([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    end_date2   = max([datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', i).group(0), "%Y-%m-%d") for i in output["tmax"].time.values])
    
    assert end_date1 - start_date1 == pd.Timedelta("0D")
    assert end_date2 - start_date2 == pd.Timedelta("0D")
    assert start_date1 - start_date2 == pd.Timedelta("0D")
    assert end_date1 - end_date2 == pd.Timedelta("0D")

    assert "ppt" in output["ppt"].time[0].values.item()
    assert "tmax" in output["tmax"].time[0].values.item()
    assert len(output["ppt"].time.values) == 1
    assert len(output["tmax"].time.values) == 1

def test_getDaymet(AOI):

    varname = ["prcp", "tmax"]
    startDate = "2018-01-01"
    endDate   = "2018-01-02"
    verbose   = True

    # Call function to get output
    output = climatePy.getDaymet(AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"prcp", "tmax"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["prcp"], xr.DataArray)
    assert isinstance(output["tmax"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["prcp"].dims == ("y", "x", "time")
    assert output["tmax"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["prcp"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["prcp"].time[1].values.item()).group(0), "%Y-%m-%d")
    start_date2 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["tmax"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date2   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["tmax"].time[1].values.item()).group(0), "%Y-%m-%d")

    assert end_date1 - start_date1 == pd.Timedelta("1D")
    assert end_date2 - start_date2 == pd.Timedelta("1D")

def test_getLivneh_monthly_case1(AOI):
    verbose   = True
    varname="wind"
    startDate="2010-01-01"
    endDate="2010-02-01"
    timeRes="monthly"

    # Call function to get output
    output = climatePy.getLivneh(AOI, varname, startDate, endDate, timeRes, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"wind"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["wind"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["wind"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[1].values.item()).group(0), "%Y-%m-%d")

    assert end_date1 - start_date1 == pd.Timedelta("29D")
    
    assert "wind" in output["wind"].time[0].values.item()
    assert len(output["wind"].time.values) == 2


def test_getLivneh_monthly_case2(AOI):

    verbose   = True
    varname="wind"
    startDate="2010-01-01"
    endDate="2010-01-01"
    timeRes="monthly"

    # Call function to get output
    output = climatePy.getLivneh(AOI, varname, startDate, endDate, timeRes, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"wind"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["wind"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["wind"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[-1].values.item()).group(0), "%Y-%m-%d")

    assert end_date1 - start_date1 == pd.Timedelta("0D")
    
    assert "wind" in output["wind"].time[0].values.item()
    assert len(output["wind"].time.values) == 1


def test_getLivneh_daily_case1(AOI):

    verbose   = True
    varname   = "wind"
    startDate = "2010-01-01"
    endDate   = "2010-01-01"
    timeRes   = "daily"

    # Call function to get output
    output = climatePy.getLivneh(AOI, varname, startDate, endDate, timeRes, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"wind"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["wind"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["wind"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[-1].values.item()).group(0), "%Y-%m-%d")

    assert end_date1 - start_date1 == pd.Timedelta("0D")
    
    assert "wind" in output["wind"].time[0].values.item()
    assert len(output["wind"].time.values) == 1

def test_getLivneh_daily_case2(AOI):

    verbose   = True
    varname   = "wind"
    startDate = "2010-01-01"
    endDate   = "2010-02-04"
    timeRes   = "daily"

    # Call function to get output
    output = climatePy.getLivneh(AOI, varname, startDate, endDate, timeRes, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"wind"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["wind"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["wind"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[-1].values.item()).group(0), "%Y-%m-%d")
    end_date2   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["wind"].time[1].values.item()).group(0), "%Y-%m-%d")

    assert end_date1 - start_date1 == pd.Timedelta("34D")
    assert end_date2 - start_date1 == pd.Timedelta("1D")
    
    assert "wind" in output["wind"].time[0].values.item()
    assert len(output["wind"].time.values) == 35

    
def test_getLivneh_fluxes_case1(AOI):

    verbose   = True
    varname   = "Baseflow"
    startDate = "2010-01-01"
    endDate   = "2010-01-02"

    # Call function to get output
    output = climatePy.getLivneh_fluxes(AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"Baseflow"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["Baseflow"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["Baseflow"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["Baseflow"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["Baseflow"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("1D")

    # check data has correct start/end dates
    assert start_date1.strftime("%Y-%m-%d") == startDate
    assert end_date1.strftime("%Y-%m-%d") == endDate

    assert "Baseflow" in output["Baseflow"].time[0].values.item()
    assert len(output["Baseflow"].time.values) == 2

def test_getLivneh_fluxes_case2(AOI):

    verbose   = True
    varname   = ["GroundHeat", "SWE"]
    startDate = "2010-01-01"
    endDate   = "2010-01-04"

    # Call function to get output
    output = climatePy.getLivneh_fluxes(AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"GroundHeat", "SWE"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["GroundHeat"], xr.DataArray)
    assert isinstance(output["SWE"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["GroundHeat"].dims == ("y", "x", "time")
    assert output["SWE"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["GroundHeat"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["GroundHeat"].time[-1].values.item()).group(0), "%Y-%m-%d")

    start_date2 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["SWE"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date2   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["SWE"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("3D")
    assert end_date2 - start_date2 == pd.Timedelta("3D")
    assert start_date2 - start_date1 == pd.Timedelta("0D")
    assert end_date2 - end_date1 == pd.Timedelta("0D")

    # check data has correct start/end dates
    assert start_date1.strftime("%Y-%m-%d") == startDate
    assert end_date1.strftime("%Y-%m-%d") == endDate
    assert start_date2.strftime("%Y-%m-%d") == startDate
    assert end_date2.strftime("%Y-%m-%d") == endDate

    assert "GroundHeat" in output["GroundHeat"].time[0].values.item()
    assert "SWE" in output["SWE"].time[0].values.item()
    
    assert len(output["GroundHeat"].time.values) == 4
    assert len(output["SWE"].time.values) == 4

def test_getLivneh_fluxes_case3(AOI):

    verbose   = True
    varname   = ["LatentHeat"]
    startDate = "2010-01-01"
    endDate   = "2010-02-02"

    # Call function to get output
    output = climatePy.getLivneh_fluxes(AOI, varname, startDate, endDate, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"LatentHeat"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["LatentHeat"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["LatentHeat"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["LatentHeat"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["LatentHeat"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("32D")

    # check data has correct start/end dates
    assert start_date1.strftime("%Y-%m-%d") == startDate
    assert end_date1.strftime("%Y-%m-%d") == endDate

    assert "LatentHeat" in output["LatentHeat"].time[0].values.item()
    
    assert len(output["LatentHeat"].time.values) == 33

def test_getPolaris_case1(AOI):
    # ---- Case 1: single variable as list ----
    verbose   = True
    varname   = ["mean alpha 5-15cm"]
    # varname = "p95 theta_s 100-200cm"
    
    # Call function to get output
    output = climatePy.getPolaris(AOI, varname, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"mean alpha 5-15cm"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["mean alpha 5-15cm"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["mean alpha 5-15cm"].dims == ("y", "x")

    # check CRS
    assert output["mean alpha 5-15cm"].crs == "EPSG:4326"

    # check dimensions
    # assert len(output["mean alpha 5-15cm"]) == 3238

    # assert output["mean alpha 5-15cm"].shape == (3238, 7078)
    
def test_getPolaris_case2(AOI):
    # ---- Case 2: single variable as string ----
    verbose   = True
    varname = "p95 theta_s 100-200cm"

    # Call function to get output
    output = climatePy.getPolaris(AOI, varname, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"p95 theta_s 100-200cm"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["p95 theta_s 100-200cm"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["p95 theta_s 100-200cm"].dims == ("y", "x")

    # check CRS
    assert output["p95 theta_s 100-200cm"].crs == "EPSG:4326"

    # check dimensions
    # assert len(output["p95 theta_s 100-200cm"]) == 3238

    # assert output["p95 theta_s 100-200cm"].shape == (3238, 7078)


def test_getPolaris_case3(AOI):
    # ---- Case 3: multiple variables ----
    verbose   = True
    varname   = ["mean clay 100-200cm", "p95 theta_s 100-200cm"]

    # TODO: problem with "mean ksat 100-200cm" varname, probably more issues with other datasets as well...
    # varname = ["mean ksat 100-200cm"]

    # Call function to get output
    output = climatePy.getPolaris(AOI, varname, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {'mean clay 100-200cm', 'p95 theta_s 100-200cm'}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["p95 theta_s 100-200cm"], xr.DataArray)
    assert isinstance(output["mean clay 100-200cm"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["p95 theta_s 100-200cm"].dims == ("y", "x")
    assert output["mean clay 100-200cm"].dims == ("y", "x")

    # check CRS
    assert output["p95 theta_s 100-200cm"].crs == "EPSG:4326"
    assert output["mean clay 100-200cm"].crs == "EPSG:4326"

    # check dimensions
    # assert len(output["p95 theta_s 100-200cm"]) == 3238
    # assert len(output["mean clay 100-200cm"]) == 3238
    
    # assert output["p95 theta_s 100-200cm"].shape == (3238, 7078)
    # assert output["mean clay 100-200cm"].shape == (3238, 7078)

def test_getMACA_month_case1(AOI):

    # ---- Case 1: single variable as list ----
    verbose   = True
    varname   = ["pr"]
    startDate = "2010-01-01"
    endDate   = "2010-02-02"
    timeRes   = "month"
    # timeRes   = 'day'
    model     = 'CCSM4'
    scenario  = 'rcp45'

    # Call function to get output
    output = climatePy.getMACA(AOI, varname, startDate, endDate,timeRes, model, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"pr"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("31D")

    startmonth = datetime.strptime(startDate, "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")
    endmonth = datetime.strptime(endDate, "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")

    # check data has correct start/end dates
    assert start_date1.strftime("%Y-%m-%d") == startmonth
    assert end_date1.strftime("%Y-%m-%d") == endmonth

    assert "pr" in output["pr"].time[0].values.item()
    
    assert len(output["pr"].time.values) == 2

def test_getMACA_month_case2(AOI):
    
    # ---- Case 1: single variable as list ----
    verbose   = True
    varname   = ["pr", "vpd"]
    startDate = "2010-01-04"
    endDate   = "2010-03-10"
    timeRes   = "month"
    # timeRes   = 'day'
    model     = 'CCSM4'
    scenario  = 'rcp45'

    # Call function to get output
    output = climatePy.getMACA(AOI, varname, startDate, endDate,timeRes, model, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"vpd", "pr"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)
    assert isinstance(output["vpd"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")
    assert output["vpd"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[-1].values.item()).group(0), "%Y-%m-%d")

    start_date2 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["vpd"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date2   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["vpd"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("59D")
    assert end_date2 - start_date2 == pd.Timedelta("59D")

    startmonth = datetime.strptime(startDate, "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")
    endmonth = datetime.strptime(endDate, "%Y-%m-%d").replace(day=1).strftime("%Y-%m-%d")

    # check data has correct start/end dates
    assert start_date1.strftime("%Y-%m-%d") == startmonth
    assert end_date1.strftime("%Y-%m-%d") == endmonth
    assert start_date2.strftime("%Y-%m-%d") == startmonth
    assert end_date2.strftime("%Y-%m-%d") == endmonth

    assert "pr" in output["pr"].time[0].values.item()
    assert "vpd" in output["vpd"].time[0].values.item()

    assert len(output["pr"].time.values) == 3
    assert len(output["vpd"].time.values) == 3

    # assert output['pr'].shape == (23, 48, 3)
    # assert output['vpd'].shape == (23, 48, 3)

def test_getMACA_day_case1(AOI):
    #  ---- Case 1: Daily ----
    verbose   = True
    varname   = ["pr"]
    startDate = "2010-01-01"
    endDate   = "2010-01-02"
    timeRes   = "day"
    model     = 'CCSM4'
    scenario  = 'rcp45'

    # Call function to get output
    output = climatePy.getMACA(AOI, varname, startDate, endDate,timeRes, model, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == { "pr"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("1D")


    assert "pr" in output["pr"].time[0].values.item()

    assert len(output["pr"].time.values) == 2

    # assert output['pr'].shape == (23, 48, 2)

def test_getMACA_day_case2(AOI):
    #  ---- Case 2: Daily multi variable ----
    verbose   = True
    varname   = ["pr", "vpd"]
    startDate = "2010-01-01"
    endDate   = "2010-01-02"
    timeRes   = "day"
    model     = 'CCSM4'
    scenario  = 'rcp45'

    # Call function to get output
    output = climatePy.getMACA(AOI, varname, startDate, endDate,timeRes, model, scenario, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"vpd", "pr"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["pr"], xr.DataArray)
    assert isinstance(output["vpd"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["pr"].dims == ("y", "x", "time")
    assert output["vpd"].dims == ("y", "x", "time")

    # Assert that the temporal resolution of the output DataArrays is correct
    start_date1 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date1   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["pr"].time[-1].values.item()).group(0), "%Y-%m-%d")

    start_date2 = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["vpd"].time[0].values.item()).group(0), "%Y-%m-%d")
    end_date2   = datetime.strptime(re.search(r'\d{4}-\d{2}-\d{2}', output["vpd"].time[-1].values.item()).group(0), "%Y-%m-%d")

    # check temporal range
    assert end_date1 - start_date1 == pd.Timedelta("1D")
    assert end_date2 - start_date2 == pd.Timedelta("1D")

    assert "pr" in output["pr"].time[0].values.item()
    assert "vpd" in output["vpd"].time[0].values.item()

    assert len(output["pr"].time.values) == 2
    assert len(output["vpd"].time.values) == 2
    
    # assert output['pr'].shape == (23, 48, 2)
    # assert output['vpd'].shape == (23, 48, 2)

def test_get3DEP_case1(AOI):
    #  ---- Case 1: 30m DEM ----
    verbose   = True
    res       = '30m'

    # Call function to get output
    output = climatePy.get3DEP(AOI, res, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"elevation"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["elevation"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["elevation"].dims == ("y", "x")

    assert output["elevation"].attrs['crs'] == "EPSG:4269"

    # assert len(output["elevation"]) == 3238
    
    # assert output['elevation'].shape == (3238, 7078)

def test_get3DEP_case2(AOI):
    #  ---- Case 2: 10m DEM ----
    verbose   = True
    res       = '10m'

    # Call function to get output
    output = climatePy.get3DEP(AOI, res, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"elevation"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["elevation"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["elevation"].dims == ("y", "x")

    assert output["elevation"].attrs['crs'] == "EPSG:4269"
    
    # assert len(output["elevation"]) == 9712
    
    # assert output['elevation'].shape == (9712, 21231)

def test_getISRIC_soils_case1(AOI):
    #  ---- Case 1: Vertisols ----
    verbose   = True
    varname       = 'Vertisols'

    # Call function to get output
    output = climatePy.getISRIC_soils(AOI, varname, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"Vertisols"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["Vertisols"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["Vertisols"].dims == ("y", "x")

    assert output["Vertisols"].attrs['crs'] == "EPSG:4326"

    # assert len(output["Vertisols"]) == 432
    
    # assert output['Vertisols'].shape == (432, 945)


def test_getISRIC_soils_case2(AOI):
    #  ---- Case 2: Gypsisols ----
    verbose   = True
    varname       = 'Gypsisols'

    # Call function to get output
    output = climatePy.getISRIC_soils(AOI, varname, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"Gypsisols"}
    
    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["Gypsisols"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["Gypsisols"].dims == ("y", "x")

    assert output["Gypsisols"].attrs['crs'] == "EPSG:4326"

    # assert len(output["Gypsisols"]) == 432
    
    # assert output['Gypsisols'].shape == (432, 945)

def test_getISRIC_soils_case3(AOI):
    # ---- Case 3: Multiple variables ----
    verbose   = True
    varname       = ['Vertisols', 'Gypsisols']

    # Call function to get output
    output = climatePy.getISRIC_soils(AOI, varname, verbose)

    # Assert that the output is a dictionary
    assert type(output) == dict

    # Assert that the output dictionary has the correct keys
    assert set(output.keys()) == {"Vertisols", "Gypsisols"}

    # Assert that the values of the output dictionary are xarray DataArrays
    assert isinstance(output["Vertisols"], xr.DataArray)
    assert isinstance(output["Gypsisols"], xr.DataArray)

    # Assert that the dimensions of the output DataArrays are correct
    assert output["Vertisols"].dims == ("y", "x")
    assert output["Gypsisols"].dims == ("y", "x")

    assert output["Vertisols"].attrs['crs'] == "EPSG:4326"
    assert output["Gypsisols"].attrs['crs'] == "EPSG:4326"

    # assert len(output["Vertisols"]) == 432
    # assert len(output["Gypsisols"]) == 432

    # assert output['Vertisols'].shape == (432, 945)
    # assert output['Gypsisols'].shape == (432, 945)
