# date and string parsing libraries
import re
from datetime import datetime, timedelta

# spatial data libraries
import geopandas as gpd
import shapely.geometry
import xarray as xr
import netCDF4 as nc
from pyproj import CRS, Proj

# data wrangling and manipulation
import numpy as np
import pandas as pd
from collections import Counter

# warnings lib
import warnings

# suppress warnings
warnings.filterwarnings('ignore', category=Warning)

def aggreg_pt_dataframes(df_list):
    """
    Process, concatenate, and pivot a list of Pandas DataFrames and return a wide DataFrame
    with a column for the 'date' and a separate column for each variable that data was retrieved for.

    This function is for internal use only and runs only if a list of dataframes is returned from 'go_get_dap_data'. 
    Handles the cases when a point is given to `dap()` and the return data results in a single grid cell.

    Args:
        out (list): A list of pandas DataFrames returned from 'go_get_dap_data' function calls.

    Returns:
        pandas.DataFrame: A wide DataFrame with a 'date' column and separate columns for each variable.
    """

    # Loop through each dataframe in df_list
    for i in range(len(df_list)):
        # print(f"i: {i}")
        
        # Column names of dataframe
        colnames = df_list[i].columns
        
        # Extract column names that are NOT the "date" column
        id_row = "".join(colnames[colnames != "date"])
        # [c for c in out[i].columns if c != "date"]
        # print(f"id_row: {id_row}")
        
        # Set column names to "date" and "value"
        df_list[i].columns = ["date", "value"]
        
        # Give the original dataset name, that was stored as a column header, as a new "name" column in the dataframe
        df_list[i]["name"] = id_row
        
        # print(f'-----------')
    
    # Concatenate the list of dataframes into a single dataframe
    df_list = pd.concat(df_list, axis=0)
    
    # Pivot data from long to wide
    df_list = df_list.pivot(index="date", columns="name", values="value")
    
    # Reset date index
    df_list = df_list.reset_index()
    
    return df_list

def shapely_to_gpd(AOI):

    """Convert a Shapely object to a GeoDataFrame.

    Args:
        AOI (shapely.geometry.base.BaseGeometry): The area of interest as a Shapely object.

    Returns:
        GeoDataFrame: A GeoDataFrame representing the area of interest.
            The GeoDataFrame has a single geometry column with an automatically assigned CRS (either EPSG:4326 or EPSG:5070)

    Note:
        The function assumes that the shapely AOI is in either WGS84 (EPSG:4326) or NAD83 (EPSG:5070) coordinate system.
    """

    # convex hull
    chull = AOI.convex_hull

    # check if convex hull is a point or a polygon
    if isinstance(chull, shapely.geometry.point.Point):
        xx, yy = chull.coords.xy
    else:
        xx, yy = chull.exterior.coords.xy

    # bounding box
    xmax = np.asarray(xx).max()
    xmin = np.asarray(xx).min()
    ymax = np.asarray(yy).max()
    ymin = np.asarray(yy).min()

    # check if AOI is in WGS84 or NAD83
    if (ymax >= -90 and ymax <= 90
        and ymin <= 90 and ymin >= -90
        and xmax >= -180 and xmax <= 180
        and xmin <= 180 and xmin >= -180):
        print("Assuming AOI CRS is EPSG:4326 (WGS84)")
        crs = CRS.from_epsg(4326)
    else:
        print("Assuming AOI CRS is EPSG:5070 (NAD83)")
        crs = CRS.from_epsg(5070)

    out = gpd.GeoDataFrame(geometry = [AOI], crs = crs)

    return out

def check_aoi(AOI):

    """Check that the AOI is in a valid format and change if needed.

    If AOI is a GeoDataFrame or GeoSeries with more than 1 geometry, converts AOI to a GeoDataFrame with a 
    single geometry representing the total bounds of all the geometries in the GeoDataFrame.

    If AOI is a shapely geometry, converts AOI to a GeoDataFrame with a single geometry representing the AOI.

    Args:
        AOI (GeoDataFrame, GeoSeries, or shapely.geometry.base.BaseGeometry): The area of interest as a GeoDataFrame, GeoSeries, or Shapely object.

    """

    # check if AOI is None
    if AOI is None:
        return None
    
    # geodataframe check
    if isinstance(AOI, (gpd.GeoDataFrame, gpd.GeoSeries)):

        if AOI.crs is None:
            raise ValueError("AOI GeoDataFrame does not have a CRS attribute. Please set a CRS attribute for the AOI GeoDataFrame.")

        # if more than one row/geometry, change geodataframe to be the total bounds of all the shapes in the geodataframe
        if len(AOI) > 1: 
            xmin, ymin, xmax, ymax = AOI.geometry.total_bounds
            AOI = gpd.GeoDataFrame(geometry=[shapely.geometry.box(xmin, ymin, xmax, ymax)], crs=AOI.crs)
            # bb = AOI.geometry.total_bounds
            # AOI = gpd.GeoDataFrame(geometry=[shapely.geometry.box(bb[0], bb[1], bb[2], bb[3])], crs=AOI.crs)

            return AOI
        
        # if single geometry and its a point, create a bounding box around the point w/ a small buffer
        if AOI.geometry.geom_type.to_list()[0] == "Point":
            xmin, ymin, xmax, ymax = AOI.buffer(0.005).geometry.total_bounds
            # xmin, ymin, xmax, ymax = AOI.geometry.total_bounds
            
            AOI = gpd.GeoDataFrame(geometry=[shapely.geometry.box(xmin, ymin, xmax, ymax)], crs=AOI.crs)

            return AOI
        
    # If AOI is a shapely geometry, convert AOI into GeoPandas dataframe 
    if isinstance(AOI, (shapely.geometry.point.Point, 
                        shapely.geometry.multipoint.MultiPoint,
                        shapely.geometry.linestring.LineString, 
                        shapely.geometry.multilinestring.MultiLineString, 
                        shapely.geometry.polygon.Polygon, 
                        shapely.geometry.multipolygon.MultiPolygon)):
            
            # convert shapely geometry to geopandas dataframe
            AOI = shapely_to_gpd(AOI)
    
    return AOI

def getExtension(x):
    """Extract the file extension from a string"""

    dot_pos = x.rfind('.')
    if dot_pos == -1:
        return ''
    else:
        return x[dot_pos+1:]
    
def get_var_dims(obj, varname = None):

    """Get the dimensions of a variable in a netcdf file"""

    with xr.open_dataset(obj, decode_times=False) as ds:

        # Get the names of the variables
        if varname is None:
            var_names = ds.data_vars.keys()
        else:
            var_names = [varname]

        # empty dataframe to concatenate onto
        empty_df = pd.DataFrame()

        for idx, val in enumerate(var_names):

            # get xyt attributes from netcdf object
            xyt = ds[val].attrs['dimensions'].split(' ')

            # insert variable name to start of list
            xyt.insert(0, val)

            # make dataframe
            empty_df = pd.concat([empty_df, pd.DataFrame([xyt], columns=['variable', 'X', 'Y', 'T'])])
        
        ds.close()
    
    return empty_df

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

def match_date_abbr(key):
    
    """Match the abbreviated time units to the full time units
    
    Args:
        key (str): The abbreviated time unit.
    
    Returns:    
        str: The full time unit.
    """

    # map of abbreviated time units to full time units
    abbr_map = {
        "H": "hours",
        "D": "days",
        "min": "minutes",
        "S": "seconds",
        "MS": "months"
        }
    
    if key in abbr_map.keys():
        return abbr_map[key]
    else:
        return key

def match_long_time_units(key):
        
        long_map = {
            "hour": "H",
            "hours": "H",
            "hourly": "H",
            "day": "D",
            "days": "D",
            "daily": "D",
            "minutes": "min",
            "seconds": "S",
            "month": "M",
            "months": "MS",
            "monthly": "MS",
            "years": "Y",
            "yearly": "Y",
            "annual": "Y",
            "annualy": "Y",
            "annually": "Y"
            }
        
        if key in long_map.keys():
            return long_map[key]
        else:
            return key

def days_to_intervals(key):

    """ Convert days to other common intervals """

    # if interval is "2-4 day"
    if key > 1 and key < 3:
        return f"{key} day"
    
    day_conversions = {
        1: "1 day",
        4: "5 days",
        5: "5 days",
        6: "5 days",
        7: "1 week",
        28: "1 month",
        29: "1 month",
        30: "1 month",
        31: "1 month",
        60: "2 months",
        90: "3 months",
        120: "4 months",
        150: "5 months",
        180: "6 months",
        365: "1 year",
        730: "2 year",
        1095: "3 years",
        1825: "5 years",
        3650: "10 years",
    }
    if key in day_conversions:
        return day_conversions[key]
    else:
        return False
    
def seconds_to_intervals(key):

    """ Convert seconds to other common intervals """

    sec_conversions = {
        1: "1 second",
        60: "1 minute",
        3600: "1 hour",
        86400: "1 day",
        432000: "5 days",
        2592000: "1 month",
        31536000: "1 year"
    }

    if key in sec_conversions:
        return sec_conversions[key]
    else:
        return False
    
def has_hms(timesteps):
    """Check if datetime object has valid hours, minutes, or seconds, or if it is a duplicate value that can be dropped"""
    
    # convert timedelta object to string
    ts = [str(i) for i in timesteps]

    # get the time differences between each timestep
    res = set(dstring.split("T")[1] for dstring in ts if "T" in dstring)

    # convert timedelta object to string
    if len(res) == 0:
        return False
    elif len(res) == 1:
        return "duplicates"
    else:
        return True

def convert_to_days(time_interval):
    # time_interval = "720 minutes"
    
    # Define conversion factors
    conversion_factors = {
        "second": 1 / 86400,
        "seconds": 1 / 86400,
        "minute": 1 / 1440,
        "minutes": 1 / 1440,
        "hour": 1 / 24,
        "hours": 1 / 24,
        "day": 1,
        "days": 1,
        "week": 7,       
        "weeks": 7,    
        "pentad": 5,      
        "pentads": 5, 
        "month": 30,
        "months": 30,
        "year": 365,
        "years": 365
    }
    
    # get time and unit parts
    parts = time_interval.split()

    if len(parts) != 2:
        return "Invalid input"
    
    # convert the value to a float and the unit to lowercase
    value, unit = float(parts[0]), parts[1].lower()
    
    # Check if the unit is in conversion_factors map
    if unit not in conversion_factors:
        return "Invalid unit"
    
    # calc the equivalent value in days
    value_in_days = value * conversion_factors[unit]

    # check if the value is a whole number and return it without decimal places
    # if value_in_days.is_integer():
    if value_in_days.is_integer() or abs(value_in_days - round(value_in_days)) < 0.10:
        return str(int(value_in_days)) + " days"
    
    # return str(int(value_in_days)) + " days"
    return f"{value_in_days:.1f} days"

def get_time_intervals(timesteps):
    """Get timedelta intervals from a list of datetime objects
    
    Args:
        timesteps (list): np.array of np.datetime64 objects

    Returns:
        np.array: numpy.timedelta64 intervals
    """

    # timesteps  = np.array([np.datetime64('2020-01-01'),np.datetime64('2020-01-02'),
    #                         np.datetime64('2020-01-03'),
    #                         np.datetime64('2020-01-04')])
    # timesteps  = np.array([np.datetime64('2020-01-01'),np.datetime64('2020-02-01'),
    #                         np.datetime64('2020-03-01'), np.datetime64('2020-04-01'),
    #                         np.datetime64('2020-05-01')])
    # timesteps = np.array([np.datetime64('2023-01-01 00:00:00'),np.datetime64('2023-01-01 01:00:00'),
    #                     np.datetime64('2023-01-01 02:00:00'),
    #                     np.datetime64('2023-01-01 03:00:00') ])
    # timesteps = np.array([np.datetime64('2023-01-01'), np.datetime64('2023-01-05'),
    #                 np.datetime64('2023-01-10'),np.datetime64('2023-01-15')])

    # timesteps = time_steps

    # check if valid hour minutes seconds in date
    valid_hms = has_hms(timesteps)

    # if valid_hms == "duplicates" then the timesteps are duplicates 
    # and can be dropped because it is just an irrelevent HMS for each timestamp
    if valid_hms == "duplicates":
        # convert HMS to days
        timesteps = timesteps.astype('datetime64[D]')

    # get the time differences between each timestep
    time_diffs = [timesteps[i+1] - timesteps[i] for i in range(len(timesteps)-1)]

    # IF no values in time_diffs
    if len(time_diffs) == 0:
        # return 0 and give warning if no values in time_diffs
        # print("Warning: time_steps has only one value. Setting dT to 0")
        return 0

    return time_diffs

def count_time_intervals(timesteps):
    """Count the number of unique time intervals in a numpy array of timedelta objects

    Args:
        timesteps (np.array): np.array of np.timedelta64 objects
    Returns:
        dict: dictionary of unique time intervals and their counts
        """

    # timesteps  = np.array([np.datetime64('2020-01-01'),np.datetime64('2020-01-02'),
    #                         np.datetime64('2020-01-03'),
    #                         np.datetime64('2020-01-04')])

    # convert timedelta object to string
    time_diffs = [str(i) for i in timesteps]

    # determine the most common interval
    interval_count = Counter(time_diffs)

    return interval_count

def validate_time_interval(time_interval):
    """Correct time interval string to a common interval

    Typically function will take a time interval string with seconds and 
    convert it to a common interval. For example, "86400 seconds" will become "1 day".

    Args:
        time_interval (str): time interval string in the format "<time_value> <time_unit>" (e.g. "86400 seconds", "1 day", "1 month")

    Returns:
        str: corrected time interval string in the format "<time_value> <time_unit>" 
    """

    # extract the time interval integer value and unit string
    time_val = int(time_interval.split(" ")[0])
    unit = time_interval.split(" ")[-1]

    # if the unit value is "seconds" or "second", check if a common interval equists and convert it
    if unit in ["seconds", "second"]:

        # convert seconds to other common intervals
        sec_convert = seconds_to_intervals(time_val)

        # if a conversion exists, convert the time interval
        if sec_convert:
            # if verbose: 
            #     print(f"Converting {time_val} seconds to {sec_convert}")
            time_interval = sec_convert

    # # if the unit value is "days" or "day", check if a common interval equists and convert it
    # if unit in ["days", "day"]:
    #     # convert days to other common intervals
    #     day_convert = days_to_intervals(time_val)

    #     # if a conversion exists, convert the time interval
    #     if day_convert:
    #         if verbose: 
    #             print(f"Converting {time_val} days to {day_convert}")
    #         time_interval = day_convert

    # return the full string if only_units is False
    return time_interval

### VERSION 2 of _resource_time() function, dealing with Livneh issues 
#### because of time interval being 0 (only a single timestamp in resource)

def _resource_time(
        nc     = None,
        T_name = None
        ):
    
    """Get time information from a xarray netcdf file
    
    Args:
        nc (xarray.core.dataset.Dataset): netcdf file to extract time informations from
        T_name (str, optional): name of the time variable. Defaults to None.
    
    Returns:
        dict: dictionary of time information with duration, interval, and nT keys
    """

    # Time variable info
    T_var_info = nc[T_name]
    
    # time variable units
    T_units    = T_var_info.attrs["units"].split(" ")[0]

    # time variable values
    time_steps = xr.decode_cf(nc)[T_name].values

    # check if time steps are isoformatted
    # if isinstance(time_steps[0], cftime._cftime.DatetimeNoLeap):
    if not isinstance(time_steps[0], (datetime, np.datetime64)):
        vec_isoformat = np.vectorize(lambda x: datetime.fromisoformat(x.isoformat()))

        # apply the vectorized function to the dates array to get an array of isoformatted date strings
        time_steps = vec_isoformat(time_steps)

    # get the time intervals between each datetime in the time_steps array
    dT = get_time_intervals(time_steps)

    # Check that there is more than one time step (dT NOT equal to 0):
    # Conditions breakdown:
        # IF dT IS 0 (first 'if' condition), then the time interval is 0 (i.e. a dataset that has a single timestamp), 
        # thus we will set the 'interval' variable to 0 and proceed on
        # OTHERWISE (the else condition), if dT is NOT 0, then we will get the time interval info 
        # for the timestamps/dataset (most common interval, verify correct units, check if monthly data, etc.)

    # Case when there is only a single timestamp in the dataset
    if not dT:
        # if interval length is 0, set interval to 0
        interval = "0"

        # print("No time intervals found")

    # Case when there are multiple timestamps in the dataset (typical case)
    else: 

        # print(f"Found {len(dT)} time intervals")

        # get the number of time intervals
        interval_count = count_time_intervals(dT)

        # get the interval values
        interval_vals = [int(i.split(' ')[0]) for i in list(interval_count.keys())]

        # get the interval units
        interval_units = [i.split(' ')[-1] for i in list(interval_count.keys())]

        # get the most common interval value
        most_freq = interval_count.most_common(1)[0][0]

        # correct seconds to days if necessary
        most_freq = validate_time_interval(most_freq)

        # get the most common interval unit
        good_units = most_freq.split(' ')[-1]

        # switch T_units with good_units IF T_units does NOT equal the most common interval unit (good_units), set T_units to the most common interval unit
        if T_units != good_units:
            T_units = good_units

        # (np.datetime64(int(most_freq.split(' ')[0]), "D")).astype('datetime64[s]').astype(int)/86400
        # (np.datetime64(most_freq.split(' ')[0]).astype('datetime64[s]').astype(int)/86400).astype(int).tolist()
        # (time_steps[0].astype('datetime64[s]').astype(int)/86400).astype(int).tolist()

        # # value in days
        # day_vals = (g["value"].values.astype('timedelta64[s]').astype(int)/86400).astype(int).tolist()
        
        # if the time interval is a month, make a dictionary with 1 month time intervals, 
        #   otherwise, use most common interval and corrected units
        if len(interval_vals) > 1 and all(i in [28, 29, 30, 31] for i in interval_vals):
            time_dict = {
                "value": 1,
                "interval": "months"
                }
        else:
            # get the most common interval
            time_dict = {
                "value": int(most_freq.split(' ')[0]),
                "interval": good_units
                }
            
        # create a string of the interval
        interval = str(time_dict["value"]) + " " + time_dict["interval"]

        # format the interval string
        interval = format_units(interval, T_units)
    
    # determine the max date of the time steps
    # If time is within 5 days of today then we call the range open
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

    # format_date(str(time_steps.min())) + "/" + (".." if maxDate == ".." else format_date(str(maxDate)))
        
    # create a dictionary of the time attributes
    time_dict = {
        "duration" : format_date(str(time_steps.min())) + "/" + (".." if maxDate == ".." else format_date(str(maxDate))),
        "interval" : interval,
        "nT"       : nT
        }

    return time_dict

# TODO: this is duplicate code with the function above, just useful for taking in a URL string INSTEAD of the actual xarray NetCDF object in _resource_time()
def _url_to_resource_time(
        URL    = None,
        T_name = None
        ):
    
    """Get time information from a xarray netcdf file URL
    
    Args:
        URL (str): url of the netcdf file to extract time informations from
        T_name (str, optional): name of the time variable. Defaults to None.
    
    Returns:
        dict: dictionary of time information with duration, interval, and nT keys
    """

    # nc     = ds
    # T_name = raw["T_name"][0]

    with xr.open_dataset(URL, decode_times=False, decode_cf = True, decode_coords = True) as nc:
        
        # Time variable info
        T_var_info = nc[T_name]
        
        # time variable units
        T_units    = T_var_info.attrs["units"].split(" ")[0]

        # time variable values
        time_steps = xr.decode_cf(nc)[T_name].values

        # check if time steps are isoformatted
        # if isinstance(time_steps[0], cftime._cftime.DatetimeNoLeap):
        if not isinstance(time_steps[0], (datetime, np.datetime64)):
            vec_isoformat = np.vectorize(lambda x: datetime.fromisoformat(x.isoformat()))

            # apply the vectorized function to the dates array to get an array of isoformatted date strings
            time_steps = vec_isoformat(time_steps)

        # # determine the max date of the time steps
        # # If time is within 5 days of today then we call the range open
        # maxDate = np.where((time_steps.max() >=  np.datetime64(datetime.now() - timedelta(days=5)) and 
        #                     time_steps.max() <=  np.datetime64(datetime.now() + timedelta(days=1))),
        #                     "..",
        #                     str(time_steps.max())
        #                     ).item()
        
        # # get length of the timesteps or if open, a None value
        # if maxDate == "..":
        #     nT = None
        # else:
        #     nT = len(time_steps)

        # get the time intervals between each datetime in the time_steps array
        dT = get_time_intervals(time_steps)

        # Check that there is more than one time step (dT NOT equal to 0):
        # Conditions breakdown:
            # IF dT IS 0 (first 'if' condition), then the time interval is 0 (i.e. a dataset that has a single timestamp), 
            # thus we will set the 'interval' variable to 0 and proceed on
            # OTHERWISE (the else condition), if dT is NOT 0, then we will get the time interval info 
            # for the timestamps/dataset (most common interval, verify correct units, check if monthly data, etc.)

        # Case when there is only a single timestamp in the dataset
        if not dT:
            # if interval length is 0, set interval to 0
            interval = "0"

            print("No time intervals found")

        # Case when there are multiple timestamps in the dataset (typical case)
        else: 
            print(f"Found {len(dT)} time intervals")

            # get the number of time intervals
            interval_count = count_time_intervals(dT)

            # get the interval values
            interval_vals = [int(i.split(' ')[0]) for i in list(interval_count.keys())]

            # get the interval units
            interval_units = [i.split(' ')[-1] for i in list(interval_count.keys())]

            # get the most common interval value
            most_freq = interval_count.most_common(1)[0][0]

            # correct seconds to days if necessary
            most_freq = validate_time_interval(most_freq)

            # get the most common interval unit
            good_units = most_freq.split(' ')[-1]

            # switch T_units with good_units IF T_units does NOT equal the most common interval unit (good_units), set T_units to the most common interval unit
            if T_units != good_units:
                T_units = good_units

            # (np.datetime64(int(most_freq.split(' ')[0]), "D")).astype('datetime64[s]').astype(int)/86400
            # (np.datetime64(most_freq.split(' ')[0]).astype('datetime64[s]').astype(int)/86400).astype(int).tolist()
            # (time_steps[0].astype('datetime64[s]').astype(int)/86400).astype(int).tolist()

            # # value in days
            # day_vals = (g["value"].values.astype('timedelta64[s]').astype(int)/86400).astype(int).tolist()
            
            # if the time interval is a month, make a dictionary with 1 month time intervals, 
            #   otherwise, use most common interval and corrected units
            if len(interval_vals) > 1 and all(i in [28, 29, 30, 31] for i in interval_vals):
                time_dict = {
                    "value": 1,
                    "interval": "months"
                    }
            else:
                # get the most common interval
                time_dict = {
                    "value": int(most_freq.split(' ')[0]),
                    "interval": good_units
                    }
                
            # create a string of the interval
            interval = str(time_dict["value"]) + " " + time_dict["interval"]

            # format the interval string
            interval = format_units(interval, T_units)
        
        # determine the max date of the time steps
        # If time is within 5 days of today then we call the range open
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

        # format_date(str(time_steps.min())) + "/" + (".." if maxDate == ".." else format_date(str(maxDate)))
            
        # create a dictionary of the time attributes
        time_dict = {
            "duration" : format_date(str(time_steps.min())) + "/" + (".." if maxDate == ".." else format_date(str(maxDate))),
            "interval" : interval,
            "nT"       : nT
            }
        
        # nc.close()
        
    return time_dict

# def _resource_time(
#         nc     = None,
#         T_name = None
#         ):
#     """Get information about the time variable in a NetCDF dataset
#     Args:
#         nc (xarray.DataArray): xarray DataArray from a NetCDF file.
#         T_name (str): The name of the time variable.

#     Returns:
#         dict: A dictionary containing the duration, interval, and number of time steps (nT).
#             - duration: The start and end dates of the time variable, formatted as "start_date/end_date".
#             - interval: The interval between time steps, formatted as "value units".
#             - nT: The number of time steps.
#     """
#     # Time variable info
#     T_var_info = nc[T_name]

#     # time variable units
#     T_units    = T_var_info.attrs["units"].split(" ")[0]
    
#     # time steps
#     time_steps = xr.decode_cf(nc)[T_name].values

#     # if time unit is seconds, infer the frequency and 
#     # check if inferred frequency is days ("D") if so, force the 'T_unit' value to days
#     if T_units in ["second", "seconds"]:

#         # infer the frequency of the time steps
#         inferred_freq = pd.infer_freq(time_steps)

#         # if the inferred frequency is NOT seconds, force the time unit to appropriate date interval 
#         if inferred_freq != "S":
#             T_units = match_date_abbr(inferred_freq)

#     # check if time steps are isoformatted
#     # if isinstance(time_steps[0], cftime._cftime.DatetimeNoLeap):
#     if not isinstance(time_steps[0], (datetime, np.datetime64)):
#         vec_isoformat = np.vectorize(lambda x: datetime.fromisoformat(x.isoformat()))

#         # apply the vectorized function to the dates array to get an array of isoformatted date strings
#         time_steps = vec_isoformat(time_steps)

#     # time_steps = xr.decode_cf(nc).time.values

#     # change in time dT
#     dT = time_steps[1:] - time_steps[:-1]

#     if len(dT) == 0:
#         print("Warning: time_steps has only one value. Setting dT to 0")
#         dT = time_steps - time_steps
        
#     # grid of time steps and units
#     g = pd.Series(dT).sort_values().value_counts().to_frame().reset_index()

#     # rename columns
#     g.columns = ["value", "n"]
    
#     # insert interval (units) column between value and n count
#     g.insert(g.columns.get_loc('value')+1, 'interval', T_units)  # insert new column
    
#     # sort by days value
#     g = g.sort_values(by='value')
    
#     # value in days
#     day_vals = (g["value"].values.astype('timedelta64[s]').astype(int)/86400).astype(int).tolist()
    
#     # if the time interval is a month, set the interval to month
#     if len(g) > 1 and all(i in [28, 29, 30, 31] for i in day_vals):
#     # if len(g) > 1 and np.isin([28, 29, 30, 31], day_vals).all():

#         g = pd.DataFrame({"value": [1], "interval": ["months"]})

#     else:
#         # get the most common interval
#         g = g.iloc[[np.argmax(g["n"])], :]

#     # time_steps.max() >= np.datetime64(datetime.now() - timedelta(days=5))
#     # time_steps.max() <=  np.datetime64(datetime.now() + timedelta(days=1))

#     # If time is within 5 days of today then we call the range Open
#     maxDate = np.where((time_steps.max() >=  np.datetime64(datetime.now() - timedelta(days=5)) and 
#                         time_steps.max() <=  np.datetime64(datetime.now() + timedelta(days=1))),
#                         "..",
#                         str(time_steps.max())
#                         ).item()
    
#     # get length of the timesteps or if open, a None value
#     if maxDate == "..":
#         nT = None
#     else:
#         nT = len(time_steps)

#     # get length of the timesteps or if open, a None value
#     # nT = np.where(maxDate == "..", None, len(time_steps)).item()
    
#     # create a string of the interval
#     interval = (g['value'].astype(str) + " " + g['interval'].astype(str)).iloc[0]

#     # format the interval string
#     interval = format_units(interval, T_units)

#     # if interval length is 0, set interval to 0
#     if len([interval]) == 0:
#         interval = "0"

#     # create a dictionary of the time attributes
#     time_dict = {
#         "duration" : format_date(str(time_steps.min())) + "/" + (".." if maxDate == ".." else format_date(str(maxDate))),
#         "interval" : interval,
#         "nT"       : nT
#         }

#     return time_dict

def _resource_grid(
        nc, 
        X_name               = None,
        Y_name               = None, 
        stopIfNotEqualSpaced = True
        ):
    
    """Extract grid information from a xarray/NetCDF

    Parameters:
        nc (xarray.DataArray): xarray DataArray from a NetCDF file.
        X_name (str): The name of the X coordinate variable. If None, it will be extracted from the NetCDF attributes.
        Y_name (str): The name of the Y coordinate variable. If None, it will be extracted from the NetCDF attributes.
        stopIfNotEqualSpaced (bool): If True, raise a warning or an exception if the grid cells are not equally spaced (default is True).

    Returns:
        pandas.DataFrame: A dataframe containing grid information, such as coordinate reference system (CRS), x and y ranges, resolution, number of columns and rows, and top-to-bottom orientation.

    """

    # if X_name is None or Y_name is missing/None
    if X_name is None or Y_name is None:
        # get X and Y names from NetCDF attributes
        atts = dap_xyzv(nc)

        X_name = omit_none(pd.unique(atts["X_name"]))
        Y_name = omit_none(pd.unique(atts["Y_name"]))

    try:
        # grid mapping dataframe
        nc_grid_mapping = pd.DataFrame.from_dict(nc["crs"].attrs, orient = "index").reset_index()
        # nc_grid_mapping = pd.DataFrame.from_dict(nc["crs"].attrs, orient = "index").reset_index()

        # rename grid mapping columns
        nc_grid_mapping.columns = ["name", "value"]
    except:
        # grid mapping dataframe with no data
        nc_grid_mapping = pd.DataFrame(columns=["name", "value"])

    # # rename grid mapping columns
    # nc_grid_mapping.columns = ["name", "value"]

    # check if degree is in units name
    if try_att(nc, X_name, "units") is not None: 
        degree  = "degree" in try_att(nc, X_name, "units").lower()

        # if degree, create proj4 string
        crs_deg = make_proj4_string(
            nc         = nc, 
            X_name     = X_name, 
            Y_name     = Y_name, 
            is_degrees = degree
            )
        
    else:
        degree = False
    
    # check if any information in grid mapping
    if len(nc_grid_mapping) == 0:
        if degree:
            print("No projection information found.\n"
            "Coordinate variable units are degrees so,\n"
            "assuming EPSG:4326")
            # crs = "EPSG:4326"
            # crs = crs_deg + ' +init=epsg:4326' 
            crs = Proj("EPSG:4326", preserve_units=True).to_proj4()
        else: 
            warnings.warn("No projection information found in nc file.")
            # crs = crs_deg + ' +init=epsg:3857' 
            
            # crs = Proj("EPSG:3857").to_proj4()
            # crs = "+proj=lcc +lat_1=25 +lat_2=60 +x_0=0 +y_0=0 +units=m +lat_0=42.5 +lon_0=-100 +a=6378137 +f=0.00335281066474748 +pm=0 +no_defs"
            crs = None
    else: 
        try:
            crs = CRS.from_cf(nc.crs.attrs).to_wkt()
            # proj_crs = crs = CRS.from_cf(nc.crs.attrs).to_proj4()
            # print(f"crs: {crs}")
            # print(f"proj_crs: {proj_crs}")
        except Exception as e:
            crs = None
            print(f"An exception occurred: {str(e)}")
        if isinstance(crs, Exception):
            crs = None
    
    # get ncols/nrows from netcdf
    ncols = nc[X_name].shape[0]
    nrows = nc[Y_name].shape[0]

    # ncols = nc[X_name].attrs["_ChunkSizes"]
    # nrows = nc[Y_name].attrs["_ChunkSizes"]

    # if exception is raised when getting X_name values
    try: 
        # X_name values
        xx = nc[X_name].values
    except Exception as e:
        xx = None
        print(f"An exception occurred: {str(e)}")

    # if an exception is raised
    if isinstance(xx, Exception):
        xx = list(range(1, ncols+1))

    # subset array removing the last and the first element, then subtract the subarrays
    rs = xx[:-1] - xx[1:]

    # Check if the minimum and maximum values of rs are equal within the given tolerance
    are_equal = np.isclose(np.min(rs), np.max(rs), rtol = 0, atol = 0.025 * abs(np.min(rs)))
    
    # if not all values are equal within tolerance
    if not are_equal:
        # if stopIfNotEqualSpaced is True
        if stopIfNotEqualSpaced:
            # throw warning
            warnings.warn("cells are not equally spaced; you should extract values as points")
        else:
            # print a warning
            raise Exception("cells are not equally spaced; you should extract values as points")
    
    # if xx values in degrees and any values are above 180 degrees, subtract 360 from xx
    if any(xx > 180) and degree:
        xx = xx -360

    # min/max x values
    xrange = [xx.min(), xx.max()]

    # x resolution
    resx = (xrange[1] - xrange[0])/(ncols - 1)

    # X1 value, first value in xx array
    X1 = xx[0]
    
    # Xn value, last value in xx array
    Xn = xx[-1]

    # if exception is raised when getting X_name values
    try: 
        # X_name values
        yy = nc[Y_name].values
    except Exception as e:
        yy = None
        print(f"An exception occurred: {str(e)}")

    # if an exception is raised
    if isinstance(yy, Exception):
        yy = list(range(1, nrows+1))

    # subset array removing the last and the first element, then subtract the subarrays
    rs = yy[:-1] - yy[1:]

    # Check if the minimum and maximum values of rs are equal within the given tolerance
    are_equal = np.isclose(np.min(rs), np.max(rs), rtol = 0, atol = 0.025 * abs(np.min(rs)))
    
    # if not all values are equal within tolerance
    if not are_equal:
        # if stopIfNotEqualSpaced is True
        if stopIfNotEqualSpaced:
            # throw warning
            warnings.warn("cells are not equally spaced; you should extract values as points")
        else:
            # print a warning
            raise Exception("cells are not equally spaced; you should extract values as points")
    
    # min/max y values
    yrange = [yy.min(), yy.max()] 

    # y resolution
    resy = (yrange[1] - yrange[0])/(nrows - 1)

    # Y1 value, first value in yy array
    Y1 = yy[0]

    # Yn value, last value in yy array
    Yn = yy[-1]

    # check make sure first yy array value is greater than last yy array value, otherwise set toptobottom to True
    if yy[0] > yy[-1]:
        toptobottom = False
    else:
        toptobottom = True

    df = pd.DataFrame({
        'crs': crs,
        'X1': X1,
        'Xn': Xn,
        'Y1': Y1,
        'Yn': Yn,
        'resX': resx,
        'resY': resy,
        'ncols': ncols,
        'nrows': nrows,
        'toptobottom': toptobottom
        },
        index = [0]
        )
    
    return df


def read_dap_file(
        URL = None, 
        varname = None, 
        var_spec = None,
        var_spec_long = None,
        id = None, 
        varmeta = True, 
        stopIfNotEqualSpaced = True
        ):
    
    """
    Read data from an OpenDAP landing page.

    Parameters:
        URL (str or list): The URL(s) of the OpenDAP landing page(s) to read data from.
        varname (str or list): The name(s) of the variable(s) to extract from the OpenDAP dataset. If None, all variables will be extracted.
        var_spec (str or list): The variable specification(s) used to extract the data. Should match the variable(s) in varname.
        var_spec_long (str or list): The long variable specification(s) used to extract the data. Should match the variable(s) in varname.
        id (str): An identifier to associate with the extracted data.
        varmeta (bool): If True, extract variable metadata along with the data (default is True).
        stopIfNotEqualSpaced (bool): If True, stop execution if the grid spacing of the dataset is not equal (default is True).

    Returns:
        pandas.DataFrame: A dataframe containing the extracted data along with associated metadata.

    """

    # check if URL is a list
    if isinstance(URL, list):
        pass
    else:
        URL = [URL]
    # empty list to append to
    raw_list = []

    # test links
    # i = "http://thredds.northwestknowledge.net:8080/thredds/dodsC/agg_met_pr_1979_CurrentYear_CONUS.nc"
    # idx = 1
    
    for idx, i, in enumerate(URL):
        if varmeta:
            print(f'Extracting resource: {i}')
            print(f'read_dap_file var_spec: {var_spec[idx]}')
            print(f'read_dap_file var_spec_long: {var_spec_long[idx]}')

        with xr.open_dataset(i, decode_times=False, decode_cf = True, decode_coords = True) as ds:
            
            if varname is not None:
                if isinstance(varname, list):
                    x = varname[idx]
                else:
                    x = varname
            else:
                x = varname
            
            # get raw data
            raw = dap_xyzv(
                ds       = ds,
                varname  = x, 
                varmeta       = varmeta,
                var_spec      = var_spec[idx],
                var_spec_long = var_spec_long[idx]
                )
            
            raw["URL"] = i
            raw["id"] = id

            res = _resource_time(
                nc     = ds, 
                T_name = raw["T_name"][0]
                )
            
            # add id variable to join on
            res["id"] = id

            # join raw dimensions and resource time data
            raw = pd.merge(
                raw, 
                pd.DataFrame(res, index = [0]), 
                on = "id"
                )
            
            X_name = raw["X_name"][0]
            Y_name = raw["Y_name"][0]

            res_grid = _resource_grid(
                nc                   = ds, 
                X_name               = X_name, 
                Y_name               = Y_name,
                stopIfNotEqualSpaced = True
                )
            
            # add id variable to join on
            res_grid["id"] = id

            # join raw data with 
            raw = pd.merge(raw, res_grid, on = "id")

            # close dataset file
            ds.close()
        
        raw_list.append(raw)

    # concatenate the list of raw dataframes into a single dataframe
    raw_concat = pd.concat(raw_list)
    
    return raw_concat