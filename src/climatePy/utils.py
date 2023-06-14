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

def format_date(date_str):

    """Format date string to YYYY-MM-DD"""

    # if input date is already in correct format, return input date_str
    if len(date_str) == 10 and re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    try:
        # regular expression match YYYY-MM-DD 
        match = re.match(r'(\d{4}-\d{2}-\d{2}(T|\s)\d{2}:\d{2}:\d{2})', date_str)

        # format first date group to YYYY-MM-DD-HH-MM_SS
        if 'T' in match.group(0):
            dt = datetime.strptime(match.group(0), '%Y-%m-%dT%H:%M:%S')
        else:
            dt = datetime.strptime(match.group(0), '%Y-%m-%d %H:%M:%S')

        # format date to YYYY-MM-DD string
        fdate = dt.strftime('%Y-%m-%d')

        return fdate
    
    except Exception:
        # if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        #     return date_str
        return ""
    
def format_units(
        interval_str = None, 
        unit_str     = None
        ):
    
    """Format interval unit string to remove duplicate units"""

    if interval_str is not None and unit_str is not None:
        # split interval into string
        words = interval_str.split()

        # remove all but the first instance of unit_str
        while words.count(unit_str) > 1:
            words.remove(unit_str)

        interval_str = " ".join(words)

    return interval_str

def get_bounding_box(nc, X_name, Y_name):

    """Get bounding box coordinates from a netCDF file"""

    lon_min = nc[X_name].values.min()
    lon_max = nc[X_name].values.max()
    lat_min = nc[Y_name].values.min()
    lat_max = nc[Y_name].values.max()

    # Create a dictionary with bounding box coordinates
    bbox = {
        'minx': lon_min, 
        'miny': lat_min, 
        'maxx': lon_max,
        'maxy': lat_max
        }
    
    return bbox

def make_proj4_string(nc, X_name, Y_name, is_degrees=True):

    """Make a proj4 string from a netCDF file"""

    # Get bounding box coordinates
    bbox = get_bounding_box(nc, X_name, Y_name)

    # Create proj4 string
    if is_degrees:
        proj_str = '+proj=longlat +datum=WGS84 +no_defs '
    else:
        proj_str = '+proj=merc +datum=WGS84 +no_defs '

    proj_str += f"+bounds={bbox['minx']},{bbox['miny']},{bbox['maxx']},{bbox['maxy']}"

    return proj_str

# # Define the projection parameters
# a = semi_major_axis  # in meters
# rf = inverse_flattening  # reciprocal of flattening
# lon_0 = prime_meridian_longitude  # in degrees
# # Create the CRS string
# crs_string = (
#     f"+proj=longlat +a={a} +rf={rf} +lon_0={lon_0} +datum=WGS84 +no_defs"
# )
# # Create the pyproj CRS object
# crs = pyproj.CRS.from_string(crs_string)
# import xarray as xr

def find_attrs(ds, varname, attributes_of_interest):
    """
    Search for specific attributes within the given variable in the provided xarray dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to search.
    varname : str
        The name of the variable to search within.
    attributes_of_interest : list of str
        A list of attribute names to search for within the variable.

    """
    # Get the variable from the dataset
    var = ds[varname]

    # Iterate over all attributes of the variable
    for attr_name in var.attrs:
        attr_value = var.attrs[attr_name]

        # Check if the attribute is in the list of attributes of interest
        if attr_name in attributes_of_interest:
            print(f'{varname}: {attr_name}={attr_value}')

def find_attrs2(ds, attributes_of_interest):
    """
    Search for specific attributes within the provided xarray dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to search.
    attributes_of_interest : list of str
        A list of attribute names to search for within the dataset.

    Returns:
    --------
    None
    """
    # Iterate over all variables in the dataset
    for varname in ds.variables:
        var = ds[varname]

        # Iterate over all attributes of the variable
        for attr_name in var.attrs:
            attr_value = var.attrs[attr_name]

            # Check if the attribute is in the list of attributes of interest
            if attr_name in attributes_of_interest:
                print(f'{varname}: {attr_name}={attr_value}')

    # Iterate over all attributes of the dataset itself
    for attr_name in ds.attrs:
        attr_value = ds.attrs[attr_name]

        # Check if the attribute is in the list of attributes of interest
        if attr_name in attributes_of_interest:
            print(f'{attr_name}={attr_value}')

# find_attrs2(ds = nc, attributes_of_interest = ["false_northing", "false_easting",
#                                             "longitude_of_central_meridian", "grid_mapping_name",
#                                                 "semi_major_axis", "inverse_flattening",
#                                                 "longitude_of_prime_meridian", "grid_mapping",
#                                                 "crs", "standard_parallel"]
#                                                 )

def get_attrs(ds, attributes_of_interest):
    """
    Search for specific attributes within the provided xarray dataset and return a dictionary of the variable attributes.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset to search.
    attributes_of_interest : list of str
        A list of attribute names to search for within the dataset.

    Returns:
    --------
    dict
        A dictionary of the variable attributes, keyed by variable name.
    """
    attributes = {}

    # Iterate over all variables in the dataset
    for varname in ds.variables:
        var = ds[varname]

        var_attributes = {}
        # Iterate over all attributes of the variable
        for attr_name in var.attrs:
            attr_value = var.attrs[attr_name]

            # Check if the attribute is in the list of attributes of interest
            if attr_name in attributes_of_interest:
                var_attributes[attr_name] = attr_value

        if var_attributes:
            attributes[varname] = var_attributes

    # Iterate over all attributes of the dataset itself
    dataset_attributes = {}
    for attr_name in ds.attrs:
        attr_value = ds.attrs[attr_name]

        # Check if the attribute is in the list of attributes of interest
        if attr_name in attributes_of_interest:
            dataset_attributes[attr_name] = attr_value

    if dataset_attributes:
        attributes['dataset'] = dataset_attributes

    return attributes

def dap_xyzv(ds, varname = None, varmeta = False, var_spec = None, var_spec_long = None):

    """Get XYTV data from DAP URL

    Args:
        ds (xarray.Dataset): The dataset object containing the data variables.
        varname (str, optional): The name of the variable to retrieve. If None, all variables in the dataset will be retrieved. Default is None.
        varmeta (bool, optional): Flag indicating whether to print variable metadata during processing. Default is False.
        var_spec (list, optional): A list of variable names used to filter the retrieved variables. Only variables with names in var_spec will be included. Default is None.
        var_spec_long (list, optional): A list of variable names used to filter the retrieved variables. Only variables with names in var_spec_long will be included. Default is None.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the retrieved variable names, their corresponding X, Y, and T dimensions, and the dimension order.
    """

    #  Get the names of the variables
    if varname is None:
        var_names = list(ds.data_vars.keys())
    else:
        var_names = [varname]

    # if var_spec is not None, filter var_names
    if var_spec is not None or var_spec_long is not None:
        var_names = [i for i in var_names if i in var_spec or i in var_spec_long]
        # var_names = [i for i in var_names if i in var_spec]
    
    # empty dataframe to concatenate onto
    raw_df = pd.DataFrame()

    # get number of variables
    n = len(var_names)

    # loop through variables
    for idx, val in enumerate(var_names):

        if varmeta:
            print(f'dap XYZV: {val} ({idx+1}/{n})')
        # val= "lambert_conformal_conic"
        # ds[val].dims
        # tmp = "time lat lon".split(" ")
        try:
            # get xyt attributes from netcdf object
            xyt = ds[val].attrs['dimensions'].split(' ')
        except:
            xyt = list(ds[val].dims)[::-1]

        # insert variable name to start of list
        xyt.insert(0, val)

        # make dataframe
        # raw_df = pd.concat([raw_df, pd.DataFrame([xyt])])
        raw_df = pd.concat([raw_df, pd.DataFrame([xyt], columns=['variable', 'X', 'Y', 'T'])])

    # if varname is None, set varname to name found in NetCDF
    if varname is None:
        varname = raw_df['variable'][0]

    # get dimension name indexes
    try:
        # get xyt attributes from netcdf object
        o = ds[varname].attrs['dimensions'].split(' ')[::-1]    
    except:
        o = ds[varname].dims
        # o = ds[varname].dims[::-1]
    
    # get time variable name
    time_var = ds[varname].dims[0]
    
    # Find the column numbers of the search strings in the first row of the DataFrame
    o = [[row.values.tolist().index(s) if s in row.values else None for s in o] for _, row in raw_df.iterrows()] 
    o = [i for i in raw_df.iloc[:, [i for sub in o for i in sub]].columns]
    
    # rename raw dataframe columns 
    raw_df.columns = ["varname", "X_name", "Y_name", "T_name"]

    # add dimension order to dataframe
    raw_df["dim_order"] = "".join([str(i) for sub in o for i in sub])
    
    # replace the time variable name with the name found in NetCDF
    raw_df.at[0, 'T_name'] = time_var

    return raw_df

def _resource_time(
        nc     = None,
        T_name = None
        ):
    """Get information about the time variable in a NetCDF dataset
    Args:
        nc (xarray.DataArray): xarray DataArray from a NetCDF file.
        T_name (str): The name of the time variable.

    Returns:
        dict: A dictionary containing the duration, interval, and number of time steps (nT).
            - duration: The start and end dates of the time variable, formatted as "start_date/end_date".
            - interval: The interval between time steps, formatted as "value units".
            - nT: The number of time steps.
    """
    # Time variable info
    T_var_info = nc[T_name]

    # time variable units
    T_units    = T_var_info.attrs["units"].split(" ")[0]
    
    # time steps
    time_steps = xr.decode_cf(nc)[T_name].values

    # check if time steps are isoformatted
    # if isinstance(time_steps[0], cftime._cftime.DatetimeNoLeap):
    if not isinstance(time_steps[0], (datetime, np.datetime64)):
        vec_isoformat = np.vectorize(lambda x: datetime.fromisoformat(x.isoformat()))

        # apply the vectorized function to the dates array to get an array of isoformatted date strings
        time_steps = vec_isoformat(time_steps)

    # time_steps = xr.decode_cf(nc).time.values

    # change in time dT
    dT = time_steps[1:] - time_steps[:-1]

    if len(dT) == 0:
        print("Warning: time_steps has only one value. Setting dT to 0")
        dT = time_steps - time_steps
        
    # grid of time steps and units
    g = pd.Series(dT).sort_values().value_counts().to_frame().reset_index()

    # rename columns
    g.columns = ["value", "n"]
    
    # insert interval (units) column between value and n count
    g.insert(g.columns.get_loc('value')+1, 'interval', T_units)  # insert new column
    
    # sort by days value
    g = g.sort_values(by='value')
    
    # value in days
    day_vals = (g["value"].values.astype('timedelta64[s]').astype(int)/86400).astype(int).tolist()
    
    # if the time interval is a month, set the interval to month
    if len(g) > 1 and all(i in [28, 29, 30, 31] for i in day_vals):
    # if len(g) > 1 and np.isin([28, 29, 30, 31], day_vals).all():

        g = pd.DataFrame({"value": [1], "interval": ["months"]})

    else:
        # get the most common interval
        g = g.iloc[[np.argmax(g["n"])], :]

    # time_steps.max() >= np.datetime64(datetime.now() - timedelta(days=5))
    # time_steps.max() <=  np.datetime64(datetime.now() + timedelta(days=1))

    # If time is within 5 days of today then we call the range Open
    maxDate = np.where((time_steps.max() >=  np.datetime64(datetime.now() - timedelta(days=5)) and 
                        time_steps.max() <=  np.datetime64(datetime.now() + timedelta(days=1))),
                        "..",
                        str(time_steps.max())
                        ).item()
    
    # get length of the timesteps or if open, a None value
    if maxDate == "..":
        nT = None
    else:
        nT = len(time_steps)

    # get length of the timesteps or if open, a None value
    # nT = np.where(maxDate == "..", None, len(time_steps)).item()
    
    # create a string of the interval
    interval = (g['value'].astype(str) + " " + g['interval'].astype(str)).iloc[0]

    # format the interval string
    interval = format_units(interval, T_units)

    # if interval length is 0, set interval to 0
    if len([interval]) == 0:
        interval = "0"

    # create a dictionary of the time attributes
    time_dict = {
        "duration" : format_date(str(time_steps.min())) + "/" + (".." if maxDate == ".." else format_date(str(maxDate))),
        "interval" : interval,
        "nT"       : nT
        }

    return time_dict

