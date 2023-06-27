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
# AOI    = gpd.read_file('src/data/boulder_county.gpkg')
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

# -----------------------------
# ---- getTerraClimNormals ----
# -----------------------------

def getTerraClimNormals(
        AOI       = None,
        varname   = None,
        month     = [i for i in range(1, 13)],
        scenario  = '19812010', 
        verbose   = False
        ):
    
    """Get Terra Climate Normals for an Area of Interest

    These layers from TerraClimate were creating using climatically aided interpolation of monthly anomalies 
    from the CRU Ts4.0 and Japanese 55-year Reanalysis (JRA-55) datasets with WorldClim v2.0 climatologies.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to be extracted
        month (int, list): Month(s) to extract data for
        scenario (str): Scenario to extract data for
        verbose (bool): Print out additional information
    
    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """

    # make sure month is a list
    if isinstance(month, int):
        month = [month]

    # collect raw meta data
    raw = climatepy_filter.climatepy_filter(
        # params    = load_data(), 
        id        = "terraclim_normals", 
        AOI       = AOI, 
        varname   = varname, 
        scenario  = scenario
        )

    # collect varname in correct order
    varname = raw.varname.unique().tolist()

    # collect duration string
    durs = raw['duration'].iloc[0].split("/")
    
    # start and end dates for dap call
    start_date = durs[0]
    end_date   = durs[1]

    # convert to datetime objects and create date range
    dates = pd.date_range(start=start_date, end=end_date, freq='MS').tolist()

    # select dates that match month input
    dates = [i for i in dates if i.month in month]

    # call dap
    dap_data = dap.dap(
        catalog   = raw, 
        AOI       = AOI, 
        startDate = min(dates).strftime("%Y-%m-%d"),
        endDate   = max(dates).strftime("%Y-%m-%d"), 
        varname   = varname,
        verbose   = verbose
        )

    return dap_data


# terrnorms = getTerraClimNormals(
# 	AOI=AOI, 
# 	varname=["tmax", "tmin"],
# 	month = 6,
# 	verbose=True
# 	)

# terrnorms['tmax'].plot()
# plt.show()

# terrnorms['tmin'].plot()
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
#     startDate = "2023-06-01",
#     endDate = "2023-06-02"
# 	verbose=True
# 	)
# gridmet = getGridMET(
# 	AOI=AOI, 
# 	varname = "pr",
#     startDate = "2023-06-01",
#     endDate = "2023-06-02",
# 	verbose=True
# 	)
# gridmet.keys()
# gridmet['pr'].isel(time=0).plot()
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

# daymet = getDaymet(
# 	AOI=AOI, 
# 	varname = ["prcp", "tmax"],
# 	startDate="2015-01-01",
# 	endDate="2016-02-20",
# 	verbose=True
# 	)
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

# ------------------
# ---- getPRISM ----
# ------------------
def getPRISM(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        timeRes   = None,
        verbose   = False
        ):
    
    """Get PRISM data for an Area of Interest
        Currently only monthly data is available.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        timeRes (str): time resolution of data to be downloaded. Options are "daily" or "monthly". Default is "monthly"
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """

    # TODO: setup timeRes argument so daily or monthly data can be got 
    # # # check timeRes argument
    if timeRes not in ['daily', 'monthly']:
        raise ValueError("timeRes must be monthly or daily")

    if timeRes is None:
        # set timeRes argument
        timeRes = "monthly"

    if timeRes == "monthly":
        # get matching arguments for climatepy_filter function
        dap_meta = dap.climatepy_dap(
            AOI       = AOI, 
            id        = "prism_monthly",
            # id        = "prism_" + timeRes, 
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
    
    elif timeRes == "daily":
        dap_data = dap.get_prism_daily(
            AOI       = AOI,
            varname   = varname,
            startDate = startDate,
            endDate   = endDate,
            verbose   = verbose
            )
        
        return dap_data
    
    return dap_data

# --------------------
# ---- getLivneh ----
# --------------------

def getLivneh(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        timeRes   = "daily",
        verbose   = False
        ):
    
    """Get Livneh Climate Data for an Area of Interest

    This dataset provides Livneh model output data as gridded estimates of daily weather parameters for North America.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        timeRes (str): Time resolution. Options include: "daily" or "monthly". Default is "daily".
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """

    if timeRes == "daily":
        # get matching arguments for climatepy_filter function
        dap_meta = dap.climatepy_dap(
            AOI       = AOI, 
            id        = "Livneh_daily", 
            varname   = varname, 
            startDate = startDate, 
            endDate   = endDate,
            verbose   = verbose
            )
    else: 
        # get matching arguments for climatepy_filter function
        dap_meta = dap.climatepy_dap(
            AOI       = AOI,
            id        = "Livneh_monthly", 
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

# livneh_month = getLivneh(
# 	AOI=AOI, 
# 	varname="wind",
# 	startDate="2010-01-01",
# 	endDate="2010-02-01", 
# 	timeRes="monthly",
# 	verbose=True
# 	)

# livneh_month.keys()
# livneh_month['wind'].isel(time=0).plot()
# plt.show()

# livneh_day = getLivneh(
# 	AOI=AOI, 
# 	varname="wind",
# 	startDate="2010-01-01",
# 	endDate="2010-01-05", 
# 	timeRes="daily",
# 	verbose=True
# 	)

# livneh_day['wind'].isel(time=0).plot()
# plt.show()
# --------------------------
# ---- getLivneh_fluxes ----
# --------------------------

def getLivneh_fluxes(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        verbose   = False
        ):
    
    """Get Livneh Fluxes Climate Data for an Area of Interest

    This dataset provides Livneh model output data as gridded estimates of daily weather parameters for North America.
    
    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """
    
    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "Livneh_fluxes", 
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

# livneh_flux = getLivneh_fluxes(
# 	AOI=AOI, 
# 	varname = "Baseflow",
# 	startDate="2010-01-01",
# 	endDate="2010-01-05", 
# 	verbose=True
# 	)
# livneh_flux.keys()
# livneh_flux['Baseflow'].isel(time=0).plot()
# plt.show()

# ----------------
# ---- getVIC ----
# ----------------

def getVIC(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        model     = 'ccsm4', 
        scenario  = 'rcp45',
        verbose   = False
        ):
    
    """Get VIC Climate Data for an Area of Interest"""

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "bcsd_vic", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        model     = model,
        scenario  = scenario,
        verbose   = verbose
        )
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# vic = getVIC(
#     AOI=AOI, 
#     model     = 'CCSM4',
#     scenario  = 'rcp45',
#     varname = "et",
#     startDate = '2004-01-01',
#     endDate   = '2004-08-05',
#     verbose   = True
#     )

# vic = getVIC(
#     AOI=AOI, 
#     varname = ["baseflow", "et"],
#     startDate = "2004-01-01",
#     endDate   = "2005-02-02",
#     model     = 'ccsm4',
#     scenario  = 'rcp45',
#     verbose   = True
#     )

# vic
# vic.keys()
# vic['et'].isel(time=0).plot()
# plt.show()
# vic['baseflow'].isel(time=0).plot()
# plt.show()


# ------------------
# ---- getNLDAS ----
# ------------------

def getNLDAS(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        model     = None, 
        verbose   = False
        ):
    
    """Get NLDAS Climate Data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Model name. Default is None.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "NLDAS", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        model     = model,
        verbose   = verbose
        )
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# nldas = getNLDAS(
# 	AOI=AOI, 
# 	startDate = '2004-01-01',
# 	endDate   = '2004-01-05',
# 	verbose   = True
# 	)

# -----------------
# ---- getMACA ---- 
# -----------------

def getMACA(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        timeRes   = 'day',
        model     = 'CCSM4', 
        scenario  = 'rcp45', 
        verbose   = False
        ):
    
    """Get MACA Climate Data for an Area of Interest

    Multivariate Adaptive Constructed Analogs (MACA) is a statistical method for downscaling Global Climate Models
    (GCMs) from their native coarse resolution to a higher spatial resolution that captures reflects 
    observed patterns of daily near-surface meteorology and simulated changes in GCMs experiments.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        timeRes (str): Time resolution. Either "mmonth" or "day". Default is 'day'.
        model (str): Model name. Default is 'CCSM4'.
        scenario (str): Scenario name. Default is 'rcp45'.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """
    
    # check timeRes argument
    if timeRes not in ['day', 'month']:
        raise ValueError("timeRes must be month or day")
    
    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "maca_" + timeRes, 
        varname   = varname, 
        model     = model,
        scenario  = scenario, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data


# maca_month = getMACA(
# 	AOI=AOI, 
# 	# varname= "pr",
# 	varname= ["pr", "tasmin"],
# 	startDate="2010-01-01",
# 	endDate="2010-03-05", 
# 	timeRes="month",
# 	# model     = 'NorESM1-M', 
# 	# scenario  = 'rcp85', 
# 	verbose=True
# 	)

# maca_month.keys()
# maca_month['air_temperature'].isel(time=0).plot()
# plt.show()
# maca_month['precipitation'].isel(time=0).plot()
# plt.show()

# maca_day = getMACA(
# 	AOI=AOI, 
# 	# varname= "pr",
# 	varname= ["tasmin", "pr", "tasmax"],
# 	startDate="2018-01-01",
# 	endDate="2018-01-15", 
# 	timeRes="day",
# 	# model     = 'NorESM1-M', 
# 	# scenario  = 'rcp85', 
# 	verbose=True
# 	)
# maca_day.keys()
# maca_day['pr'].isel(time=0).plot()
# plt.show()
# maca_day['tasmin'].isel(time=0).plot()
# plt.show()
# maca_day['tasmax'].isel(time=0).plot()
# plt.show()

# ---------------------
# ---- getWordClim ----
# ---------------------

# TODO this function throws an error on the first call and succeeds on the second call (temp file issue?) 
def getWorldClim(
        AOI       = None,
        varname   = None,
        date 	  = None,
        res       = None,
        verbose   = False
        ):
    
    """Get WorldClim data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        date (str): date in the form "YYYY-MM-DD"
        res (str): Resolution of data to download. One of: "10m", "5m", "2.5m", "30s". Default is "10m".
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """	

    if res is None:
        res = "10m"
    
    if res not in ["10m", "5m", "2.5m", "30s"]:
        raise ValueError("res must be one of: 10m, 5m, 2.5m, 30s")

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "wc2.1_" + res,
        varname   = varname,
        startDate = date,
        endDate   = date,
        verbose   = verbose
        )
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# # get matching arguments for climatepy_filter function
# dap_meta = dap.climatepy_dap(
#     AOI       = AOI,
#     id        = "wc2.1_5m",
# 	varname= ["tmax", "tavg"],
#     startDate = "1990-01-01",
#     endDate   = "1990-01-01",
#     verbose   = True
#     )
# worldclim = getWorldClim(
# 	AOI=AOI, 
# 	varname= ["tmax"],
# 	date="1990-01-01", 
# 	res = "5m",
# 	verbose=True
# 	)

# worldclim.keys()
# worldclim['tmax'].plot()
# plt.show()
# worldclim = getWorldClim(
# 	AOI=AOI, 
# 	varname= ["tmax", "tavg"],
# 	date="1990-01-01", 
# 	res = "5m",
# 	verbose=True
# 	)

# worldclim.keys()
# worldclim['tmax'].plot()
# plt.show()
# worldclim['tavg'].plot()
# plt.show()

# ---------------------
# ---- getWordClim ----
# ---------------------

def getSoilGrids(
        AOI       = None,
        varname   = None,
        date 	  = None,
        res       = None,
        verbose   = False
        ):
    
    """Get ISRIC Soil Grids data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """	

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "ISRIC Soil Grids",
        varname   = varname,
        verbose   = verbose
        )
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# vert_sg = getSoilGrids(
#     AOI=AOI,
#     varname="Vertisols",
#     verbose=True
#     )
# vert_sg.keys()
# vert_sg['Vertisols'].plot()
# plt.show()

# sand_sg = getSoilGrids(
#     AOI=AOI,
#     varname="sand_60-100cm_mean",
#     verbose=True
#     )
# sand_sg.keys()
# sand_sg['sand_60-100cm_mean'].plot()
# plt.show()

# --------------------
# ---- getUSGSDEM ----
# --------------------

def getUSGSDEM(
        AOI       = None,
        res       = None,
        verbose   = False
        ):
    
    """Get USGS Digatal Elevation Model (DEM) data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        res (str): Resolution of data to download. One of: "10m", "30m". Default is "30m".
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """	

    if res is None:
        res = '30m'

    # check res argument
    if res not in ['10m', '30m']:
        raise ValueError("timeRes must be '10m' or '30m'")

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "USGS 3DEP",
        asset     = res + " CONUS DEM",
        varname   = "elevation",
        verbose   = verbose
        )
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# usgsdem = getUSGSDEM(
#     AOI=AOI, 
#     res= "30m",
#     verbose=True
#     )

# usgsdem.keys()
# usgsdem['elevation'].plot()
# plt.show()

# --------------------
# ---- getNASADEM ----
# --------------------

def getNASADEM(
        AOI       = None,
        verbose   = False
        ):
    
    """Get NASA Digatal Elevation Model (DEM) data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """	

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "NASADEM",
        varname   = "elevation",
        verbose   = verbose
        )
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# nasadem = getNASADEM(
# 	AOI=AOI, 
# 	verbose=True
# 	)

# nasadem.keys()
# nasadem['elevation'].plot()
# plt.show()