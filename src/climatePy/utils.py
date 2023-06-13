# date and string parsing libraries
import re
from datetime import datetime, timedelta

# spatial data libraries
import geopandas as gpd
import shapely.geometry
import xarray as xr
import netCDF4 as nc

# data wrangling and manipulation
import numpy as np
import pandas as pd

# misc libraries
import warnings

# suppress warnings
warnings.filterwarnings('ignore', category=Warning)

def getExtension(x):
    """Get the file extension from a string."""

    dot_pos = x.rfind('.')
    if dot_pos == -1:
        return ''
    else:
        return x[dot_pos+1:]
    
def make_ext(cat):
    """Create a bounding box from catalog coordinates.

    Args:
        cat (pandas.Series): A Series containing catalog coordinates.

    Returns:
        shapely.geometry.box: A bounding box representing the catalog coordinates.
    """

    bbox = shapely.box(
        min(cat['Xn'], cat['X1']),
        min(cat['Yn'], cat['Y1']),
        max(cat['Xn'], cat['X1']),
        max(cat['Yn'], cat['Y1'])
        )
    # return gpd.GeoDataFrame(geometry=[bbox], crs=cat["crs"])
    return bbox

def make_vect(cat):
    """Create a vector representation of a catalog entry.

    Args:
        cat (pandas.Series): A Series containing catalog information.

    Returns:
        geopandas.GeoDataFrame: A GeoDataFrame representing the catalog entry.
    """

    return gpd.GeoDataFrame(geometry = [make_ext(cat)], crs=cat["crs"])

def omit_none(x):

    """Remove None values from list"""

    return [v for v in x if not None]

def try_att(nc, variable, attribute):
    """Get the value of a specified attribute from a NetCDF variable

    Args:
        nc (netCDF4.Dataset): The NetCDF dataset object.
        variable (str): The name of the variable from which to retrieve the attribute.
        attribute (str): The name of the attribute to retrieve.

    Returns:
        object: The value of the specified attribute if it exists, otherwise None.
    """

    try:
        return nc[variable].attrs[attribute]
    except:
        return None
    