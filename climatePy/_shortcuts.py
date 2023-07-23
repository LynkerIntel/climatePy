# Data manipulation tools
import pandas as pd
import geopandas as gpd
import xarray as xr
import shapely
from shapely.geometry import Point

#  os lib
import os

# import climatePy modules
from . import _utils as utils
from . import _dap as dap
from . import _climatepy_filter as climatepy_filter
from . import _netrc_utils as netrc_utils

# import climatePy._utils as utils
# from climatePy import params
# from climatePy import _dap as dap
# from climatePy import _climatepy_filter as climatepy_filter

# warnings lib
import warnings

# suppress warnings
warnings.filterwarnings('ignore', category=Warning)


# test data
# AOI    = gpd.read_file('climatePy/data/san_luis_obispo_county.gpkg')
# AOI    = gpd.read_file('climatePy/data/boulder_county.gpkg')

# ----------------------
# ---- getTerraClim ----
# ----------------------

def getTerraClim(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        dopar     = True,
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
        dopar (bool): Use parallel processing
        verbose (bool): Print out additional information

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """
    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
    # dap_meta = climatepy_dap(
        AOI       = AOI, 
        id        = "terraclim", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
    # dap_data = dap(
        **dap_meta
        )
    
    return dap_data

# terr = getTerraClim(AOI = AOI, 
#              varname = "tmax",
#              startDate = "2010-01-01",
#              endDate = "2010-12-31")

# -----------------------------
# ---- getTerraClimNormals ----
# -----------------------------

def getTerraClimNormals(
        AOI       = None,
        varname   = None,
        month     = [i for i in range(1, 13)],
        scenario  = '19812010', 
        dopar     = True,
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
        dopar (bool): Use parallel processing
        verbose (bool): Print out additional information
    
    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """

    # make sure month is a list
    if isinstance(month, int):
        month = [month]

    # collect raw meta data
    raw = climatepy_filter.climatepy_filter(
        # params    = params(), 
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
        dopar     = dopar,
        verbose   = verbose
        )

    return dap_data

# --------------------
# ---- getGridMET ----
# --------------------

def getGridMET(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        dopar     = True,
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
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# --------------------
# ---- getGLDAS  ----
# --------------------

# TODO: Fix netrc and dodsrc file creation process to work with earthdata credentials
def getGLDAS(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        model     = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get GLDAS Data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Model to download.
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """

    if not netrc_utils.checkNetrc():

        raise Exception("netrc file not found. Please run writeNetrc() with earth data credentials..")
    
    else:

        x = netrc_utils.writeDodsrc()

        # get matching arguments for climatepy_filter function
        dap_meta = dap.climatepy_dap(
            AOI       = AOI, 
            id        = "GLDAS", 
            varname   = varname, 
            startDate = startDate, 
            endDate   = endDate,
            verbose   = verbose
            )
        
        dap_meta['dopar'] = dopar

        # need to provide dap_meta dictionary object directly as input
        dap_data = dap.dap(
            **dap_meta
            )
        
        # remove dodsrc file
        os.unlink(x)

        return dap_data
    
# --------------------
# ---- getMODIS  ----
# --------------------


def getMODIS(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        model     = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get MODIS Data for an Area of Interest

    Args:
        AOI (geopandas dataframe, shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Model to download.
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """
    
    if not netrc_utils.checkNetrc():

        raise Exception("netrc file not found. Please run writeNetrc() with earth data credentials..")
    
    else:

        x = netrc_utils.writeDodsrc()

        # get matching arguments for climatepy_filter function
        dap_meta = dap.climatepy_dap(
            AOI       = AOI, 
            id        = "MODIS", 
            varname   = varname, 
            startDate = startDate, 
            endDate   = endDate,
            verbose   = verbose
            )
        
        dap_meta['dopar'] = dopar

        # need to provide dap_meta dictionary object directly as input
        dap_data = dap.dap(
            **dap_meta
            )
        
        # remove dodsrc file
        os.unlink(x)

        return dap_data
    
# -------------------
# ---- getDaymet ----
# -------------------

def getDaymet(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        dopar     = True,
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
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

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
        dopar     = True,
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
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# ------------------
# ---- getPRISM ----
# ------------------
def getPRISM(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        timeRes   = None,
        dopar     = True,
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
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
        
        dap_meta['dopar'] = dopar

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
        dopar     = True,
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
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# --------------------------
# ---- getLivneh_fluxes ----
# --------------------------

def getLivneh_fluxes(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        dopar     = True,
        verbose   = False
        ):
    
    """Get Livneh Fluxes Climate Data for an Area of Interest

    This dataset provides Livneh model output data as gridded estimates of daily weather parameters for North America.
    
    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

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
        dopar     = True,
        verbose   = False
        ):
    
    """Get VIC Climate Data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Climate model. Options include: 'ccsm4', 'cnrm_cm5', 'gfdl_cm3', 'hadgem2_es', 'ipsl_cm5a_lr', 'miroc_esm', 'mri_cgcm3', 'noresm1_m'
        scenario (str): Climate scenario. Options include: 'rcp45', 'rcp85'
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """

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
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# ------------------
# ---- getNLDAS ----
# ------------------

def getNLDAS(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        model     = None, 
        dopar     = True,
        verbose   = False
        ):
    
    """Get NLDAS Climate Data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Model name. Default is None.
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """
    if not netrc_utils.checkNetrc():

        raise Exception("netrc file not found. Please run writeNetrc() with earth data credentials..")
    
    else:

        x = netrc_utils.writeDodsrc()

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
        
        dap_meta['dopar'] = dopar
        
        # need to provide dap_meta dictionary object directly as input
        dap_data = dap.dap(
            **dap_meta
            )
        
        # unlink Dodsrc file
        os.unlink(x)
    
        return dap_data

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
        dopar     = True,
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
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# -------------------
# ---- getCHIRPS ----
# -------------------

# TODO: too much data is extracted when function is called, need to look into what is going on here with the dap function
# TODO: currently crashes my computer, even when I try to run on a small AOI and for a short time period
def getCHIRPS(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        dopar     = True,
        verbose   = False
        ):
    
    """Get CHIRPS data for an Area of Interest

    CHIRPS is a global dataset of daily precipitation estimates, with a spatial resolution of 0.05 degrees (~5 km).
    Currently only monthly data is available.

    Args:
        AOI (geopandas dataframe, shapely geometry): Area of Interest polygon to extract data for.
        varname (str): variable name to extract (e.g. tmin).
        startDate (str): start date of data to be downloaded (YYYY-MM-DD). Default is None.
        endDate (str): end date of data to be downloaded (YYYY-MM-DD). Default is None.
        dopar (bool): use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): print verbose output. Default is False.
    """

    timeRes = "monthly"

    # make sure timeRes is capitalized correctly
    timeRes = " ".join(word.capitalize() for word in timeRes.split())

    # Regex to capitalize first letter of each word
    # timeRes = re.sub(r"(^|[[:space:]])([[:alpha:]])",
    # 	lambda match: match.group(1) + match.group(2).upper(), timeRes)

    # correctly formatted timeRes values
    good_timeRes = ["Pentad", "Annual", "Daily", "Monthly"]

    # check timeRes argument is valid
    if timeRes not in good_timeRes:
        raise ValueError(f"timeRes must be one of: {', '.join(good_timeRes)}")

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "chirps20Global" + timeRes + "P05", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        verbose   = verbose
        )
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# -----------------
# ---- getLOCA ----
# -----------------

def getLOCA(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        model     = 'CCSM4',
        scenario  = 'rcp45',
        dopar     = True,
        verbose   = False
        ):
    
    """Get LOCA Climate Data for an Area of Interest

    LOCA is a statistical downscaling technique that uses past history to add improved fine-scale detail to global climate models.

    LOCA has been used to downscale 32 global climate models from the CMIP5 archive at a 1/16th degree spatial resolution, 
    covering North America from central Mexico through Southern Canada. The historical period is 1950-2005,
    and there are two future scenarios available: RCP 4.5 and RCP 8.5 over the period 2006-2100 (although some models stop in 2099). 
    The variables currently available are daily minimum and maximum temperature, and daily precipitation.

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        model (str): Model name. Default is 'CCSM4'.
        scenario (str): Scenario name. Default is 'rcp45'.
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data

    """

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI, 
        id        = "loca", 
        varname   = varname, 
        startDate = startDate, 
        endDate   = endDate,
        model     = model,
        scenario  = scenario,
        verbose   = verbose
        )
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# --------------------
# ---- getPolaris ----
# --------------------

def getPolaris(
        AOI       = None,
        varname   = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get Polaris Climate Data for an Area of Interest

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
        id        = "polaris", 
        varname   = varname, 
        verbose   = verbose
        )
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# ---------------------
# ---- getWordClim ----
# ---------------------

# TODO this function throws an error on the first call and succeeds on the second call (temp file issue?) 
def getWorldClim(
        AOI       = None,
        varname   = None,
        date 	  = None,
        res       = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get WorldClim data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        date (str): date in the form "YYYY-MM-DD"
        res (str): Resolution of data to download. One of: "10m", "5m", "2.5m", "30s". Default is "10m".
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# ------------------------
# ---- getISRIC_soils ----
# ------------------------
# getSoilGrids vs getISRIC_soils

def getISRIC_soils(
        AOI       = None,
        varname   = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get ISRIC Soil Grids data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# -----------------
# ---- get3DEP ----
# -----------------

def get3DEP(
        AOI       = None,
        res       = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get USGS Digatal Elevation Model (DEM) data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        res (str): Resolution of data to download. One of: "10m", "30m". Default is "30m".
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar
    
    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# --------------------
# ---- getNASADEM ----
# --------------------

def getNASADEM(
        AOI       = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get NASA Digatal Elevation Model (DEM) data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
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
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data

# -------------------------
# ---- AquaGoesSSTAnom ----
# -------------------------

def AquaGoesSSTAnom(
        AOI       = None,
        varname   = None,
        startDate = None,
        endDate   = None,
        units     = None,
        dopar     = True,
        verbose   = False
        ):
    
    """Get SST anomolies from AquaGoesSSTAnomC data for an Area of Interest

    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in "YYYY-MM-DD" format.
        endDate (str): End date in "YYYY-MM-DD" format.
        units: temperature units to return data in "C" or "F". Default is "C".
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """	

    if units == None:
        units = "C"
    
    if units not in ["C", "F"]:
        raise ValueError("units must be either 'C' or 'F'")
    
    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "aquaGoesSSTAnom" + units,
        varname   = varname,
        startDate = startDate,
        endDate   = endDate,
        verbose   = verbose
        )
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data


# ------------------
# ---- getLCMAP ----
# ------------------

def getLCMAP(
        AOI       = None,
        varname   = None,
        startDate = None, 
        endDate   = None, 
        dopar     = True,
        verbose   = False
        ):
    
    """Get LCMAP data for an Area of Interest
    
    Args:
        AOI (shapely.geometry.polygon.Polygon): Area of interest as a shapely polygon or geopandas dataframe
        varname (str, list): Variable name(s) to download.
        startDate (str): Start date in the form "YYYY-MM-DD"
        endDate (str): End date in the form "YYYY-MM-DD"
        dopar (bool): Use parallel processing. If True multiple workers will fetch data from remote sources in parallel.
        verbose (bool): Print verbose output. Default is False.

    Returns:
        dictionary of xarray.DataArray(s): xarray DataArray containing climate data
    """	

    # get matching arguments for climatepy_filter function
    dap_meta = dap.climatepy_dap(
        AOI       = AOI,
        id        = "LCMAP",
        varname   = varname,
        startDate = startDate,
        endDate   = endDate,
        verbose   = verbose
        )
    
    dap_meta['dopar'] = dopar

    # need to provide dap_meta dictionary object directly as input
    dap_data = dap.dap(
        **dap_meta
        )
    
    return dap_data
