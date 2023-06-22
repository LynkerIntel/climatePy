import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import shapely

from src.climatePy import dap, climatepy_filter, utils

import xarray as xr
import matplotlib.pyplot as plt

from shapely.geometry import Point

from src.climatePy import dap, climatepy_filter, utils

# AOI    = gpd.read_file('src/data/miami_dade_county.gpkg')
# AOI    = gpd.read_file('src/data/san_luis_obispo_county.gpkg')

# ----------------------
# ---- getTerraClim ----
# ----------------------

def getTerraClim(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        verbose   = False
        ):
    
    """Get Terra Climate Normals for an Area of Interest

    These layers from TerraClimate were creating using climatically aided interpolation of monthly anomalies from the CRU Ts4.0 
    and Japanese 55-year Reanalysis (JRA-55) datasets with WorldClim v2.0 climatologies.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to be extracted
        startDate (str): Start date for data extraction in YYYY-MM-DD format
        endDate (str): End date for data extraction in YYYY-MM-DD format
        verbose (bool): Print out additional information

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """
    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "terraclim", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# terrmulti = getTerraClim(
# 	AOI=AOI, 
# 	varname = ['tmin', "tmax", "ppt"],
# 	startDate="2010-01-01",
# 	endDate="2010-02-03", 
# 	verbose=True
# 	)

# terrmulti["tmin"].isel(time=0).plot()
# plt.show()
# terrmulti["ppt"].isel(time=0).plot()
# plt.show()
# terrmulti["tmax"].isel(time=0).plot()
# plt.show()


# --------------------
# ---- getGridMET ----
# --------------------

def getGridMET(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        verbose   = False
        ):
    
    """Get GridMet Climate Data for an Area of Interest

    gridMET is a dataset of daily high-spatial resolution (~4-km, 1/24th degree) surface meteorological data 
    covering the contiguous US from 1979-yesterday. These data are updated daily.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (list): Variable name(s) to download. Options include:
            ['pr', 'rmin', 'rmax', 'srad', 'sph', 'swe', 'tmmn', 'tmmx', 'vs', 'vpd', 'ws']
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "gridmet", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )
    
    # dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# gridmet = getGridMET(
# 	AOI=AOI, 
# 	varname = ["pr", "rmin", "rmax"],
# 	startDate="2000-01-01",
# 	endDate="2000-02-03",
# 	verbose=True
# 	)

# gridmet.keys()
# gridmet['pr'].isel(time=8).plot()
# plt.show()

# gridmet['daily_minimum_relative_humidity'].isel(time=0).plot()
# # gridmet['pr'].plot()
# plt.show()

# -------------------
# ---- getDaymet ----
# -------------------

def getDaymet(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        verbose   = False
        ):
    
    """Get Daymet Climate Data for an Area of Interest

    This dataset provides Daymet Version 4 model output data as gridded estimates of daily weather parameters for North America.
    Daymet output variables include the following parameters: minimum temperature, maximum temperature, precipitation, shortwave radiation,
    vapor pressure, snow water equivalent, and day length. The dataset covers the period from January 1, 1980 to December 31 of 
    the most recent full calendar year. Each subsequent year is processed individually at the close of a calendar year after
    allowing adequate time for input weather station data to be of archive quality.  Daymet variables are continuous surfaces provided as individual files,
    by year, at a 1-km x 1-km spatial resolution and a daily temporal resolution. Data are in a Lambert Conformal
    Conic projection for North America and are in a netCDF file format compliant with Climate and Forecast (CF) metadata conventions.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (list): Variable name(s) to download. Options include:	
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "daymet4", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

daymet = getDaymet(
	AOI=AOI, 
	varname = ["prcp", "tmax"],
	startDate="2015-01-01",
	endDate="2016-02-20",
	verbose=True
	)
# daymet.keys()
# daymet['tmax'].isel(time=9).plot()
# plt.show()

# -----------------
# ---- getBCCA ----
# -----------------

def getBCCA(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None,
        model     = 'CCSM4',
        scenario  = 'rcp45', 
        ensemble  = None,  
        verbose   = False
        ):
    
    """Get BCCA data for an Area of Interest

    This dataset provides BCCA model output data as gridded estimates of daily weather parameters for North America.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download. Options include:
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Model name. Default is 'CCSM4'.
        scenario (str): Scenario name. Default is 'rcp45'.
        ensemble (str): Ensemble name. Default is None.
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """
    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "bcca", 
        varname   = varname, 
        model     = model,
        scenario  = scenario, 
        ensemble  = ensemble, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# bcca = getBCCA(
# 	AOI=AOI, 
# 	varname="tasmax",
# 	startDate="2010-01-01",
# 	endDate="2010-01-05", 
# 	verbose=True
# 	)
# bcca.keys()
# bcca['tasmax'].isel(time=0).plot()
# plt.show()