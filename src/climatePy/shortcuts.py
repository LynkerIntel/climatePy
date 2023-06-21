import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import shapely

from src.climatePy import dap, climatepy_filter, utils

import xarray as xr
import matplotlib.pyplot as plt

from shapely.geometry import Point

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