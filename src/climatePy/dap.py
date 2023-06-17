# date and string parsing libraries
from datetime import datetime, timedelta, timezone
from dateutil.parser import parse
from dateutil import tz
import re

# spatial data libraries
import shapely
from shapely.ops import transform
from shapely.geometry import mapping
import geopandas as gpd
import xarray as xr
import rasterio as rio
import rioxarray as rxr
from pyproj import CRS, Proj
# import netCDF4 as 
import inspect

# data wrangling and manipulation
import numpy as np
import pandas as pd

# import utils from src.climatePy
from src.climatePy import climatepy_filter, utils

def parse_date(duration, interval):
        """Parse the date range based on the duration and interval.

        Args:
            duration (str): The duration string in the format "start_date/end_date".
            interval (str): The interval string specifying the time unit.
        
        Returns:
            pd.DatetimeIndex: A pandas DatetimeIndex representing the parsed date range.
        """
        
        # split duration string
        d = duration.split("/")
        
        # if end date is "..", set it to today's date
        if d[1] == "..":
                d[1] = datetime.now().strftime("%Y-%m-%d")

        # if interval in ["1 month", "1 months"]:
        if any(keyword in interval for keyword in ["1 month", "1 months", "monthly"]):
        # if any(keyword in interval for keyword in ["1 month", "1 months", "31 days", "monthly"]):
                d[0] = datetime.strptime(d[0], "%Y-%m-%d").strftime("%Y-%m-01")

        # if interval in ["hour", "hours"]:
        if any(keyword in interval for keyword in ["hour", "hours"]):
                d[0] = datetime.strptime(d[0], "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")
                d[1] = datetime.strptime(d[1], "%Y-%m-%d").strftime("%Y-%m-%d %H:%M:%S")

        # if interval is 31 days, change to 1 month
        if interval in ["31 days","31.5 days"]:
            interval = "1 month"

        # if interval is 365, 365.5 days, change to 1 year
        if interval in ["365 days","365.5 days"]:
            interval = "1 year"

        interval_map = {
            "hour": "H",
            "hours": "H",
            "minute": "min",
            "minutes": "min",
            "second": "S",
            "seconds": "S",
            "month": "MS",  # Month Start
            "months": "MS"  # Month Start
            }
        
        # split interval string
        interval_type = interval.split(" ")[-1]

        # get frequency from interval_map
        freq = interval_map.get(interval_type, interval_type[0])
        
        # # convert start_date and end_date to pandas Timestamp objects
        # start_date = pd.Timestamp(d[0])
        # end_date = pd.Timestamp(d[1])
        # # calculate the number of days between start and end dates
        # delta = (end_date - start_date) / (nT - 1)
        # # generate the date range
        # date_range = pd.date_range(start=start_date, end=end_date, freq=str(int(delta.days))+'D')

        return pd.date_range(start=d[0], end=d[1], freq=freq)

# def parse_date2(duration, nT):
    # duration = catalog["duration"].iloc[i]
    # interval = catalog["interval"].iloc[i]
    # nT = catalog["nT"].iloc[i]
#     d = duration.split("/")
    
#     if d[1] == "..":
#         d[1] = pd.Timestamp.today().strftime("%Y-%m-%d")

#     duration = pd.to_datetime(d[1]) - pd.to_datetime(d[0])
    
#     start_date = pd.to_datetime(d[0])
#     end_date = pd.to_datetime(d[1])
#     duration_in_days = (end_date - start_date).days

#     # calculate interval in days
#     interval_in_days = duration_in_days / nT
# 	interval_in_days = 31
#     # calculate the number of days to shift the frequency
#     days_to_shift = (start_date + pd.offsets.MonthBegin(1) - start_date).days
    
#     # create frequency string with the offset of days to shift
#     freq = str(int(interval_in_days)) + 'D'
#     freq = pd.tseries.offsets.DateOffset(days=days_to_shift) 
    
#     # + pd.tseries.offsets.CustomBusinessDay(n=interval_in_days)
    
#     return pd.date_range(start=start_date, end=end_date, freq=freq)

# 	interval_in_days = duration / nT
    
#     if interval_in_days >= pd.Timedelta(days=365):
#         freq = 'AS'
#     elif interval_in_days >= pd.Timedelta(days=30):
#         freq = 'M'
#     elif interval_in_days >= pd.Timedelta(days=1):
#         freq = 'D'
#     else:
#         freq = 'H'
    
#     return pd.date_range(start=d[0], end=d[1], freq=freq)

# def parse_date2(duration, interval):
#     d = duration.split("/")
    
#     if d[1] == "..":
#         d[1] = datetime.now().strftime("%Y-%m-%d")

#     # Get the start and end dates
#     start_date, end_date = map(lambda x: datetime.strptime(x, '%Y-%m-%d'), d)

#     # Set the start date to the first day of the month
#     start_date = start_date.replace(day=1)

#     if "hour" in interval:
#         # Truncate the start and end dates to the nearest hour
#         start_date = start_date.replace(minute=0, second=0)
#         end_date = end_date.replace(minute=0, second=0)
        
#     # interval_map = {
#     #     "hour": "%H",
#     #     "hours": "%H",
#     #     "minute": "%M",
#     #     "minutes": "%M",
#     #     "second": "%S",
#     #     "seconds": "%S",
#     #     "month": "MS",  # Month Start
# 	# 	"months": "MS"  # Month Start
#     # }
#     interval_map = {
#         "hour": "H",
#         "hours": "H",
#         "minute": "min",
#         "minutes": "min",
#         "second": "S",
#         "seconds": "S",
#         "month": "MS",  # Month Start
# 		"months": "MS"  # Month Start
#     }
    
#     interval_type = interval.split(" ")[-1]
#     print("interval_type: ", interval_type)
#     freq = interval_map.get(interval_type, interval_type[0])
#     print("frequency date: ", freq)
    
#     return pd.date_range(start=d[0], end=d[1], freq=freq)
#     # return pd.date_range(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"), freq=freq)
# def filter_row(i, catalog, AOI):
#     # make a bounding box from the catalog row
#     cat_box = make_ext(catalog.iloc[i])

#     # make a bounding box from the AOI and reproject to catalog crs
#     aoi_box = AOI.to_crs(catalog["crs"].iloc[0])['geometry'][0].bounds

#     # try to intersect the bounding boxes, if it fails append None to out list
#     try:
#         return shapely.box(*aoi_box).intersection(cat_box)
#     except Exception as e:
#         return None

def dap_crop(
    URL       = None,
    catalog   = None,
    AOI       = None, 
    startDate = None, 
    endDate   = None,
    varname   = None, 
    verbose   = True
    ):
    """Crop a catalog entry to a specified area of interest and time period.
    
    Args:
        URL (str, optional): The URL of the catalog entry. Defaults to None.
        catalog (pd.DataFrame, optional): The catalog dataframe. Defaults to None.
        AOI (geopandas.GeoDataFrame, optional): The area of interest for cropping. Defaults to None.
        startDate (str, optional): The start date of the time period. Defaults to None.
        endDate (str, optional): The end date of the time period. Defaults to None.
        varname (str or list, optional): The variable name(s) to filter the catalog. Defaults to None.
        verbose (bool, optional): Flag to control verbosity of progress messages. Defaults to True.
        
	Returns:
        pd.DataFrame: The cropped catalog entry.
    """
    
    # if a URL is provided, call read_dap_file
    if URL is not None:

        # stash metadata
        catvar  = catalog["variable"].values
        catmod  = catalog["model"].values
        catens  = catalog["ensemble"].values
        catsen  = catalog["scenario"].values
        catcrs  = catalog["crs"].values

        catalog = utils.read_dap_file(
            URL                  = URL, 
            varname              = varname,
            var_spec             = catalog["variable"].values,
            var_spec_long        = catalog["varname"].values,
            id                   = "local",
            varmeta              = verbose, 
            stopIfNotEqualSpaced = True
            )
        
        # add missing column to catalog
        catalog["tiled"] = ""
        catalog["variable"] = catvar
        catalog["model"]    = catmod
        catalog["ensemble"] = catens
        catalog["scenario"] = catsen

        # replace None values in col1 with a string
        catalog = catalog.fillna({'crs': catcrs[0]})

    # TIME
    if startDate is None and endDate is None:
            catalog["T"]    = "[0:1:" + (catalog['nT'] - 1).astype(int).astype(str) + "]"
            catalog["Tdim"] = catalog["nT"]
            
            tmp = [i.split("/") for i in catalog["duration"]]
            catalog = catalog.assign(startDate = [i[0] for i in tmp], endDate = [i[1] for i in tmp])
    else:
        if endDate is None:
            endDate = startDate
        
        # check if its hourly date
        if any(keyword in catalog['interval'].iloc[0] for keyword in ["hour", "hours"]):
            startDate += " 00:00:00"
            endDate   += " 23:00:00"

        # Convert startDate and endDate to pandas Timestamp objects
        startDate = pd.Timestamp(startDate)
        endDate   = pd.Timestamp(endDate)

        # out list
        out = []

        # if interval == "monthly normal" set interval column to "month", otherwise keep interval value
        catalog.loc[:, "interval"] = np.where(catalog["interval"] == "monthly normal", "month", catalog["interval"])
        # catalog.loc[:, "interval"] = np.where(catalog["interval"] == "monthly normal", "month", catalog["interval"])
        
        # loop over each row of catalog and do date parsing
        for i in range(len(catalog)): 

            if verbose:
                print(f'Parsing dates: {i+1}')

            time_steps = parse_date(
                duration = catalog["duration"].iloc[i], 
                interval = catalog["interval"].iloc[i]
                )

            if catalog['nT'].iloc[i] == 1 and not np.isnan(catalog['nT'].iloc[i]):
                        out.append(pd.concat([
                                catalog.iloc[[i], :],
                                pd.DataFrame({
                                        'T': ['[0:1:0]'],
                                        'Tdim': [1],
                                        'startDate': [time_steps[0]],
                                        'endDate': [time_steps[0]]
                                })
                        ], axis=1))
            elif startDate > max(time_steps) or endDate < min(time_steps):
                out.append(None)
                
            else:
                T1 = np.argmin(abs(time_steps - startDate))
                Tn = np.argmin(abs(time_steps - endDate))

                out.append(
                    pd.concat([catalog.iloc[[i], :].reset_index(drop=True), pd.DataFrame({
                                                                            'T': [f"[{T1}:1:{Tn}]"],
                                                                            'Tdim': [(Tn - T1) + 1],
                                                                            'startDate': [time_steps[T1]],
                                                                            'endDate': [time_steps[Tn]]
                                                                            })], axis=1))  
                                                                
            # # Concatenate out list of dataframes into single catalog dataframe
            # catalog = pd.concat(out, axis=0, ignore_index=True)

        # Concatenate out list of dataframes into single catalog dataframe
        catalog = pd.concat(out, axis=0, ignore_index=True)

        if len(catalog) == 0:
            raise ValueError(f"Requested Time not found in {set(catalog['duration'])}")
    
    # space XY
    if AOI is None:
            catalog["X"]    = "[0:1:" + (catalog['ncols'] - 1).astype(int).astype(str) + "]"
            catalog["Y"]    = "[0:1:" + (catalog['nrows'] - 1).astype(int).astype(str) + "]"
    else: 
            # if AOI is given, filter out rows not in AOI bounding box
            out = []

            # catalog = find_intersects(
            # 	catalog = catalog,
            # 	AOI     = AOI
            # 	)

            # catalog length to track progress
            n = len(catalog)

            # loop over catalog and filter out rows not in AOI bounding box
            for i in range(len(catalog)):
                
                # print messages if verbose is True
                if verbose: 
                    print(f'Filtering out rows not in AOI bounding box: ({i+1}/{n})')
                
                # make a bounding box from the catalog row
                cat_box = utils.make_ext(catalog.iloc[i])

                # make a bounding box from the AOI and reproject to catalog crs
                aoi_box = AOI.to_crs(catalog["crs"].iloc[0])['geometry'][0].bounds

                # try to intersect the bounding boxes, if it fails append None to out list
                try:
                    out.append(shapely.box(*aoi_box).intersection(cat_box))
                except Exception as e:
                    out.append(None)
            # out = Parallel(n_jobs=-1)(delayed(go_get_dap_data) (dap_data.iloc[i].to_dict()) for i in range(len(dap_data)))
            # find the indices of the None values in out and remove the corresponding rows from catalog
            drops = [i for i, x in enumerate(out) if x is None]
        
            # drop rows from catalog and remove None values from out list
            if drops:
                catalog = catalog.drop(drops)
                out     = [x for x in out if x is not None]

            # catalog length to track progress
            n = len(catalog)

            # loop over each row and do X/Y coord mapping
            for i in range(len(catalog)):
                # print messages if verbose is True
                if verbose: 
                    print(f'Mapping X/Y coords: ({i+1}/{n})')

                X_coords = np.linspace(catalog.iloc[i, catalog.columns.get_loc('X1')], catalog.iloc[i, catalog.columns.get_loc('Xn')], num = int(catalog.iloc[i, catalog.columns.get_loc('ncols')]))
                
                Y_coords = np.linspace(catalog.iloc[i, catalog.columns.get_loc('Y1')], catalog.iloc[i, catalog.columns.get_loc('Yn')],  num = int(catalog.iloc[i, catalog.columns.get_loc('nrows')]))
            
                ys = [np.argmin(np.abs(Y_coords - out[i].bounds[1])), np.argmin(np.abs(Y_coords - out[i].bounds[3]))]
                xs = [np.argmin(np.abs(X_coords - out[i].bounds[0])), np.argmin(np.abs(X_coords - out[i].bounds[2]))]
                
                catalog.loc[i, 'Y'] = f"[{':1:'.join(map(str, sorted(ys)))}]"
                catalog.loc[i, 'X'] = f"[{':1:'.join(map(str, sorted(xs)))}]"

                catalog.at[i, 'X1'] = min(X_coords[[i + 1 for i in xs]])
                catalog.at[i, 'Xn'] = max(X_coords[[i + 1 for i in xs]])
                catalog.at[i, 'Y1'] = min(Y_coords[[i + 1 for i in ys]])
                catalog.at[i, 'Yn'] = max(Y_coords[[i + 1 for i in ys]])
                catalog.at[i, 'ncols'] = abs(np.diff(xs))[0] + 1
                catalog.at[i, 'nrows'] = abs(np.diff(ys))[0] + 1
    
    # DIMENSION ORDER STRINGS
    first  = catalog['dim_order'].str[0].fillna('T').values[0]
    second = catalog['dim_order'].str[1].fillna('Y').values[0]
    third  = catalog['dim_order'].str[2].fillna('X').values[0]

    # properly format URL string for tiled data
    if catalog['tiled'].str.contains('XY').any():
        catalog['URL'] = catalog['URL'] + '?' + catalog['varname'] + catalog[first] + catalog[second] + catalog[third]
    else:
        catalog['URL'] = catalog['URL'] + '?' + catalog['varname'] + catalog[first] + catalog[second] + catalog[third]
    
    # check varname
    if varname is not None:
        # check if varname is a str, convert to list if so
        if isinstance(varname, str):
            varname = [varname]
        if varname not in catalog['varname'].unique() and varname not in catalog['variable'].unique():
            var_list = "\t > ".join(catalog['varname'].unique())
            raise ValueError(f"variable(s) in resource include:\n\t> {var_list}")
        
        catalog['varname'] == varname
        catalog =  catalog.query("varname == @varname")

    # replace NaN values with None
    catalog = catalog.replace({np.nan: "NA"})

    # TODO: look into what this does, keep commented out for now
    # catalog["X"] = None
    # catalog["Y"] = None
    # catalog["T"] = None

    return catalog


def match_args(func, *args, **kwargs):

    """Match default arguments for a function.

    This function takes a function and a variable number of positional and keyword arguments.
    It matches the provided arguments with the default arguments of the function signature
    and returns a dictionary containing the matched arguments.

    Args:
        func (callable): The function for which arguments need to be matched.
        *args: Positional arguments that need to be matched with function parameters.
        **kwargs: Keyword arguments that need to be matched with function parameters.

    Returns:
        dict: A dictionary containing the matched arguments as key-value pairs.

    """
    sig = inspect.signature(func)
    matched_args = {}
    for k, v in sig.parameters.items():
            if k in kwargs:
                    matched_args[k] = kwargs[k]
            elif args:
                    matched_args[k] = args[0]
                    args = args[1:]
            else:
                    matched_args[k] = v.default
    return matched_args

def climatepy_dap(*args, verbose = False, **kwargs):

        """ClimatePy DAP (DAP).
        
        This is an internal function that works to take varied argument inputs and appropriatly match 
        them to climatepy_filter() and dap_crop() functions. 
        Args:
            *args: Positional arguments to be matched with parameters of `climatepy_filter` and `dap_crop`.
            verbose (bool, optional): Whether to print verbose output during DAP operations. Defaults to False.
            **kwargs: Keyword arguments to be matched with parameters of `climatepy_filter` and `dap_crop`.

        Returns:
            dict: A dictionary containing the matched arguments for DAP operations.

        """

        # get matching arguments for climatepy_filter function
        matches1            = match_args(climatepy_filter.climatepy_filter, *args, **kwargs)

        # get catalog from climatepy_filter
        x = climatepy_filter.climatepy_filter(**matches1)

        # add catalog to list of arguments that'll be passed to dap
        matches1['catalog'] = x

        # matches1['verbose'] = True
        matches1['verbose'] = verbose
        matches1['varname'] = None

        # match arguments for dap_crop function
        matches2 = match_args(dap_crop, *args, **kwargs)

        # get matching arguments in climatepy_filter and dap_crop
        dap_matches = {k: matches1[k] for k in matches1 if k in matches2}

        if verbose: 
            print("dap_matches: ", dap_matches)

        return dap_matches