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
from rtree import index
import xarray as xr
import rasterio as rio
import rioxarray as rxr
from pyproj import CRS, Proj
import netCDF4 as nc
import inspect

# data wrangling and manipulation
import numpy as np
import pandas as pd

# library for parallel processing
from joblib import Parallel, delayed

# import climatePy modules
from . import _utils as utils
from . import _climatepy_filter as climatepy_filter

# from climatePy import _utils as utils
# from climatePy import _climatepy_filter as climatepy_filter

# warnings lib
import warnings

# suppress warnings
warnings.filterwarnings('ignore', category=Warning)

# from src.climatePy import climatepy_filter, utils
# from climatePy import climatepy_filter, utils
# import utils from src.climatePy
# from climatePy import _utils, climatepy_filter
# from climatePy import _utils, climatepy_filter
# from climatePy import _utils, climatepy_filter

######################################################
######################################################

def dap_crop(
    URL       = None,
    catalog   = None,
    AOI       = None, 
    startDate = None, 
    endDate   = None,
    varname   = None, 
    verbose   = False
    ):
    """Crop a catalog entry to a specified area of interest and time period.
    
    Args:
        URL (str, optional): The URL of the catalog entry. Defaults to None.
        catalog (pd.DataFrame, optional): The catalog dataframe. Defaults to None.
        AOI (geopandas.GeoDataFrame, optional): The area of interest for cropping. Defaults to None.
        startDate (str, optional): The start date of the time period. Defaults to None.
        endDate (str, optional): The end date of the time period. Defaults to None.
        varname (str or list, optional): The variable name(s) to filter the catalog. Defaults to None.
        verbose (bool, optional): Flag to control verbosity of progress messages. Defaults to Falser.
        
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

                catalog.at[i, 'X1'] = min(X_coords[[i + 1 if i + 1 < len(X_coords) else i for i in xs]])
                catalog.at[i, 'Xn'] = max(X_coords[[i + 1 if i + 1 < len(X_coords) else i for i in xs]])
                catalog.at[i, 'Y1'] = min(Y_coords[[i + 1 if i + 1 < len(Y_coords) else i for i in ys]])
                catalog.at[i, 'Yn'] = max(Y_coords[[i + 1 if i + 1 < len(Y_coords) else i for i in ys]])

                # catalog.at[i, 'X1'] = min(X_coords[[i + 1 for i in xs]])
                # catalog.at[i, 'Xn'] = max(X_coords[[i + 1 for i in xs]])
                # catalog.at[i, 'Y1'] = min(Y_coords[[i + 1 for i in ys]])
                # catalog.at[i, 'Yn'] = max(Y_coords[[i + 1 for i in ys]])

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

def dap(
        URL         = None,
        catalog     = None,
        AOI         = None,
        startDate   = None,
        endDate     = None,
        varname     = None,
        grid        = None,
        start       = None,
        end         = None,
        toptobottom = False,
        dopar       = True,
        verbose     = False
        ):

        """Get Data (Data Access Protocol)

        This function provides a consistent data access protocol (DAP) to a wide
        range of local and remote resources including VRT, TDS, NetCDF.
        
        Define and get data from a DAP resource.
        
        Parameters:
        - URL: str, optional
            Local file path or URL.
        - catalog: object, optional
            Subset of open.dap catalog.
        - AOI: object, optional
            List containing an extent() and crs.
        - startDate: object, optional
            For non "dated" items, start can be called by index.
        - endDate: object, optional
            For non "dated" items, end can be called by index.
        - varname: object, optional
            Variable name.
        - grid: object, optional
            A list containing an extent() and crs.
        - start: object, optional
            For non "dated" items, start can be called by index.
        - end: object, optional
            For non "dated" items, end can be called by index.
        - toptobottom: bool, optional
            Should data be inverse?
        - dopar: bool, if True, parallelize the download of the data 
        - verbose: bool, optional
            Should dap_summary be printed?
        
        Details:
        Wraps dap_get and dap_crop into one.
        If AOI is None, no spatial crop is executed.
        If startDate and endDate are None, no temporal crop is executed.
        If only endDate is None, it defaults to the startDate.
        
        Returns: dictionary of xarray.DataArray(s): xarray DataArray containing climate data
        """
    

        if not isinstance(toptobottom, bool):

            # print("Checking if all toptobottom values are Nan...")

            # convert to float to check for nan
            nan_chk = toptobottom.astype(float)

            # if all toptobottom values are Nan, then make toptobottom False 
            if np.isnan(np.sum(nan_chk)):

                # print("--> setting toptobottom to False")
                toptobottom = False

        # else:
        # 	print("toptobottom is already a boolean")
        # 	print(f"toptobottom = {toptobottom}")

        # check to make sure URL or catalog is provided
        if URL is None and catalog is None:
            raise ValueError("URL or catalog must be provided")
        

        # check if a single string, if so, make a list of string
        if isinstance(URL, str):
            URL = [URL]

        # if URL is None then only use catalog URL column
        if URL is None:
            URL = catalog.URL.values.tolist()

        else:
            # convert Numpy array to list
            url_lst = catalog.URL.values.tolist()

            # extend list with URL
            url_lst.extend(URL)

            # set URL to list of URLS
            URL = url_lst
            
        # # If AOI is a shapely geometry, convert AOI into geodataframe
        # if isinstance(AOI, (shapely.geometry.point.Point, 
        #                     shapely.geometry.multipoint.MultiPoint,
        #                     shapely.geometry.linestring.LineString, 
        #                     shapely.geometry.multilinestring.MultiLineString, 
        #                     shapely.geometry.polygon.Polygon, 
        #                     shapely.geometry.multipolygon.MultiPolygon)):
            
        #     # convert shapely geometry to geopandas dataframe
        #     AOI = utils.shapely_to_gpd(AOI)

        # check that AOI meets requirements, if a shapely geometry the AOI is transformed into a geodataframe
        AOI = utils.check_aoi(AOI)
            
        # check if "vrt" or "tif" in URL list, or if "vsi" in all of URL list
        if any([utils.getExtension(i) in ['vrt', "tif"] for i in URL]) or all(["vsi" in i for i in URL]):

            if verbose:
                print("Getting VRT/TIF data")
                
            # get the vrt catalog features for each URL
            vrt_data = vrt_crop_get(
                URL         = URL,
                catalog     = catalog,
                AOI         = AOI,
                grid        = grid,
                varname     = varname,
                start       = start,
                end         = end,
                toptobottom = toptobottom,
                verbose     = False
                )
            
            # # get the vrt catalog features for each URL
            # vrt_data = vrt_crop_get2(
            #     URL         = URL,
            #     catalog     = catalog,
            #     AOI         = AOI,
            #     grid        = grid,
            #     varname     = varname,
            #     start       = start,
            #     end         = end,
            #     toptobottom = toptobottom,
            #     verbose     = verbose
            #     )
            
            return vrt_data

        else:
            if verbose:
                print("Getting DAP data")

            # get the dap catalog features for each URL
            dap_data = dap_crop(
                URL       = URL,
                catalog   = catalog,
                AOI       = AOI,
                startDate = startDate,
                endDate   = endDate,
                varname   = varname,
                verbose   = False
                )
            
            # if dopar:
            #     if verbose:
            #         print("Getting DAP data in parallel")
            # else:
            #     if verbose:
            #         print("Getting DAP data in serial")

            # get dap data
            dap_data = dap_get(
                dap_data = dap_data,
                dopar    = dopar,
                verbose  = False
                )
            
            return dap_data
        
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
        matches1 = match_args(climatepy_filter.climatepy_filter, *args, **kwargs)

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

def var_to_da(var, dap_row):

    """Converts a variable to an xarray DataArray with coordinate reference system (CRS).

    Args:
        var (numpy.ndarray): The variable to be converted to a DataArray.
        dap_row (pandas.Series): A pandas Series containing metadata information.
        
    Returns:
        xarray.DataArray: The variable converted to a DataArray with CRS included.

    """
    
    # var = get_data(dap_row)
    # dap_row = dap_row

    # dates = pd.to_datetime(dates)
    dates = pd.date_range(
        start   = dap_row['startDate'], 
        end     = dap_row['endDate'], 
        periods = dap_row['Tdim']
        )

    # concatenate variable name with date and model info
    name = dap_row['variable'] + '_' + dates.strftime('%Y-%m-%d-%H-%M-%S') + '_' + dap_row['model'] + '_' + dap_row['ensemble'] + '_' + dap_row['scenario']
    name = name.str.replace('_NA', '', regex=False)
    name = name.str.replace('_na', '', regex=False)

    # extract variable name
    vars = dap_row['variable']

    # if variable name is NA, use varname
    if len(vars) == 0:
        vars = dap_row['varname']

    # clean up timeseries names
    names_ts = "_".join([vars, dap_row["model"], dap_row["ensemble"], dap_row["scenario"]])
    names_ts = names_ts.replace("_NA", "")
    names_ts = names_ts.replace("_na", "")
    names_ts = names_ts.replace("__", "_")
    names_ts = names_ts.rstrip("_")

    # # if dap_row has 1 column and 1 row, or 1 key/value
    # if len(dap_row.keys()) == 1 and len(dap_row.values()) == 1:
    #     # reshape var into a 2D array
    #     var_2d = var.reshape((len(dates), -1))

    #     # create a dictionary of column names and values
    #     var_dict = {f'var_{i}': var_2d[:, i] for i in range(var_2d.shape[1])}
    #     var_dict = {key.replace("var_", f'{names_ts}_'): var_dict[key] for key in var_dict.keys()}

    #     # create a DataFrame with dates and var_dict as columns
    #     df = pd.DataFrame({'date': dates, **var_dict})

    # if dap_row has 1 column and 1 row, or 1 key/value
    if dap_row['ncols'] == 1 and dap_row['nrows'] == 1:
        # reshape var into a 2D array
        # var_2d = var.reshape((len(dates), -1))
 
        # create a dictionary of column names and values
        # var_dict = {f'var_{i}': var_2d[:, i] for i in range(var_2d.shape[1])}
        # var_dict = {key.replace("var_", f'{names_ts}_'): var_dict[key] for key in var_dict.keys()}

        # create a DataFrame with dates and var_dict as columns
        # df = pd.DataFrame({'date': dates, **var_dict})
        # create a DataArray with dates and the variable array as the other column
        df = pd.DataFrame({'date': dates, names_ts:np.squeeze(var.values)})
        
        return df

    # x resolution
    resx = (dap_row['Xn'] - dap_row['X1'])/(dap_row['ncols'] - 1)

    # y resolution
    resy = (dap_row['Yn'] - dap_row['Y1'])/(dap_row['nrows'] - 1)

    xmin = dap_row['X1'] - 0.5 * resx
    xmax = dap_row['Xn'] + 0.5 * resx
    ymin = dap_row['Y1'] - 0.5 * resy
    ymax = dap_row['Yn'] + 0.5 * resy

    # expand dimensions of 'var' if it's a 2D array
    if var.ndim == 2:
        var = np.expand_dims(var, axis=-1)
    
    # check if size of first dimension of 'var' is equals number of rows in 'dap'
    if var.shape[2] != dap_row["nrows"] or dap_row["nrows"] == dap_row["ncols"]:
    # if var.shape[2] != dap_row["nrows"]:
        # transpose the first two dimensions of 'var' if not in correct order
        var = var.transpose(dap_row["Y_name"], dap_row["X_name"], dap_row["T_name"])
        # var2 = var.transpose(dap_row["T_name"], dap_row["X_name"],  dap_row["Y_name"])
    
    # Create the DataArray with the CRS included
    r = xr.DataArray(
        var,
        coords = {
            'y': np.linspace(ymax, ymin, dap_row['nrows'], endpoint=False),
            # 'y': -1*np.linspace(ymin, ymax, dap_row['nrows'], endpoint=False),
            'x': np.linspace(xmin, xmax, dap_row['ncols'], endpoint=False),
            'time': dates,
            'crs': dap_row['crs']
            },
            dims=['y', 'x', 'time']
            )

    # if toptobottom is True, flip the data vertically
    if dap_row['toptobottom']:

        # vertically flip each 2D array
        flipped_data = np.flip(r.values, axis=0)
        # flipped_data = np.flipud(r.values)

        # create new xarray DataArray from flipped NumPy array
        r = xr.DataArray(
            flipped_data,
            dims   = ('y', 'x', 'time'),
            coords = {'y': r.y, 'x': r.x, 'time': r.time}
            )

    # set the name attribute of the DataArray
    r = r.assign_coords(time=name)

    return r

def get_data(dap_row):
    """Internal function for retrieving data from a DAP (Data Access Protocol) source.

    Args:
        dap_row (pandas.Series): A pandas Series containing metadata information.

    Returns:
        xarray.DataArray: The retrieved variable data.

    """

    ds = xr.open_dataset(f"{dap_row['URL']}#fillmismatch")

    var = ds[dap_row['varname']]

    ds.close()

    return var

def go_get_dap_data(dap_row):

    """Internal function for retrieving DAP (Data Access Protocol) data and converting it to a DataArray.

    Args:
        dap_row (pandas.Series): A pandas Series containing metadata information.

    Returns:
        xarray.DataArray or str: The retrieved data as a DataArray, or the original URL if an error occurred.

    """

    # dap_row = dap_data.iloc[i].to_dict()
    try:
        if "http" in dap_row["URL"]:
            x = var_to_da(var = get_data(dap_row), dap_row = dap_row)
        else:
            raise Exception("dap_to_local() not avaliable, yet, dataset URL must be in http format")
    except Exception as e:
        return dap_row["URL"]
    
    return x

def add_varname_attr(
        out = None, 
        dap_data = None, 
        verbose = False
        ):
    
    for da, varb in zip(out, dap_data["variable"]):
        if verbose: 
            print(f'Adding "variable" attribute {varb} to DataArray')
        da.attrs["variable"] = varb

    for da, varname in zip(out, dap_data["varname"]):
        if verbose: 
            print(f'Adding "var" attribute {varname} to DataArray')
        da.attrs["varname"] = varname

def merge_across_time(data_arrays, verbose = False):

    """Merge DataArrays across time"""

    if verbose:
        print("Merging DataArrays across time")

    # create a dictionary to store DataArrays for each unique variable_name
    da_dict = {}

    n = len(data_arrays)

    for idx, val in enumerate(data_arrays):
        variable_name = val.attrs.get('variable', 'unknown')
        # print("Iterating through list of DataArrays: ", variable_name, "-", idx+1, "/", len(data_arrays), " DataArrays")

        if verbose:
            print(f"Iterating through list of DataArrays: {variable_name} - ({idx+1}/{n})")
            # print("Iterating through list of DataArrays: ", variable_name)

        if variable_name not in da_dict:
            if verbose:
                print("----> variable ", variable_name, "NOT IN da_dict")
                # print("variable NOT IN da_dict adding dataarray list with key: ", variable_name)
            da_dict[variable_name] = [val]
        else:
            if verbose:
                print("----> variable ", variable_name, "IN da_dict")
                # print("variable IN da_dict, appending data array to key: ", variable_name)
            da_dict[variable_name].append(val)

    # for da in data_arrays:
    #     variable_name = da.attrs.get('variable_name', 'unknown')
    #     if variable_name not in da_dict:
    #         da_dict[variable_name] = [da]
    #     else:
    #         da_dict[variable_name].append(da)

    # concatenate DataArrays for each unique variable_name
    concat_da = []
    for variable_name, da_list in da_dict.items():

        if verbose:
            print("--> concatenating time dims: ", variable_name)

        # concatenate DataArrays along the time dimension
        cda = xr.concat(da_list, dim='time')

        # add variable_name as an attribute to the concatenated DataArray
        cda.attrs['variable'] = variable_name

        # add the concatenated DataArray to the list of output DataArrays
        concat_da.append(cda)

    return concat_da

def dap_get(dap_data, dopar = True, varname = None, verbose = False):
    
    """Get DAP resource data.

    Args:
        dap_data (pandas.DataFrame): A DataFrame containing metadata information for DAP resources.
        dopar (bool, optional): Flag indicating whether to perform parallel execution. Defaults to True.
        varname (str, optional): The variable name to filter the DAP resources. Defaults to None.
        verbose (bool, optional): Flag indicating whether to print verbose output. Defaults to False.

    Returns:
        dict or xarray.DataArray: A dictionary of DataArrays or a single DataArray, representing the retrieved DAP data.

    Raises:
        ValueError: If the provided varname is not found in the DAP resources.

    """
    
    # check if varname is in dap_data
    if varname is not None:
        if varname not in dap_data["varname"].values and varname not in dap_data["variable"].values:
        # if varname not in dap_data['varname'].unique() and varname not in dap_data['variable'].unique():
            errstr = "\t > ".join(dap_data["varname"].unique())
            raise ValueError(f'variable(s) in resource include:\n\t > {errstr}' )
        
        # filter done dap_data to only include varname if varname is None
        dap_data = dap_data[
            (dap_data.get("varname", False) == varname) |
            (dap_data.get("variable", False) == varname)
            ]
        
    if dopar:
        # make go_get_dap_data calls in parallel
        out = Parallel(n_jobs=-1)(delayed(go_get_dap_data) (dap_data.iloc[i].to_dict()) for i in range(len(dap_data)))
        # out_lst = [go_get_dap_data(dap_row = dap_data.iloc[i].to_dict()) for i in range(len(dap_data))]
    else:
        # store output list 
        out = []
        # get number of rows in dap_data
        n = len(dap_data)
        # # next is to loop over each row in dap_data and call go_get_dap_data
        for i in range(len(dap_data)):
            if verbose:
                print(f'Getting dap data: ({i+1}/{n})')

            x = go_get_dap_data(dap_row = dap_data.iloc[i].to_dict())
            out.append(x)

    # If out returns a list of dataframes (typically because a single point was given as the AOI),
    # then process the list of dataframes into a single dataframe and return it (timeseries data of the point)
    if isinstance(out[0], pd.core.frame.DataFrame):
        
        out = utils.aggreg_pt_dataframes(out)

        return out
    
    # add variable name attribute to each DataArray in the output list
    add_varname_attr(
        out      = out,
        dap_data = dap_data, 
        verbose  = verbose
        )
    
    # merge across time
    out = merge_across_time(data_arrays = out, verbose = verbose)
    
    # concatenated_da = xr.concat(data_arrays, dim=('time', "variable_name"))

    # TODO: USE THIS LIST COMPREHENSION, standard for loop is easier for debugging
    # next is to loop over each row in dap_data and call go_get_dap_data
    # out = [go_get_dap_data(dap_data.iloc[x].to_dict()) for x in range(len(dap_data))]
    # out = [go_get_dap_data(dap_data.iloc[x]) for x in range(len(dap_data))]

    # get data names
    out_names = list(dict.fromkeys([name.replace("_$", "") for name in dap_data['variable'].tolist()]))
    # out_names = list(dict.fromkeys([name.replace("_$", "") for name in dap_data['varname'].tolist()]))
    # out_names = list(set([name.replace("_$", "") for name in dap_data["varname"]]))
    # out_names = [name.replace("_$", "") for name in dap_data["varname"]]
    # out_names = dap_data['varname'].str.replace('_$', '').tolist()

    # put list and names into a dictionary
    out = dict(zip(out_names, out))

    # Check if first DataArray is not a SpatRaster
    if not isinstance(out[next(iter(out))], xr.core.dataarray.DataArray):
        if verbose:
            print("Processing non DataArray data...")

        # out = list(out.values())
        # out = xr.merge(out, join="outer")
        # out = reduce(lambda dtf1, dtf2: dtf1.merge(dtf2, on="date", how="outer"), out)

        return out
    
    elif any(dap_data["tiled"].str.contains("XY")):
        if verbose:
            print("Processing 'XY' data...")

        ll = {}
        u = np.unique([da.units for da in out])

        if len(u) == 1:
            out = xr.combine_by_coords(out)
            out["units"] = (["layer"], [u[0]] * len(out["layer"]))
            ll[dap["varname"].iloc[0]] = out
            out = ll

            return out
        else:
            # out = dict(zip(out_names, out))
            return out

    elif any(dap_data["tiled"] == "T"):

        print("Processing 'T' data...")

        # out = xr.concat(out, dim="time").sortby("time")
        return out
    else:
        if verbose:
            print("Processing as normal...")
        
        return out

    return out

def var_to_rast(var, dap_row):
    """
    Convert a variable to a raster format.

    Args:
        var (numpy.ndarray): The variable data to convert.
        dap_row (pandas.Series): A row from the Data Access Protocol (DAP) climate catalog pandas dataframe containing metadata information.

    Returns:
        xr.DataArray: The converted variable as a DataArray.

    """
    
    # dates = pd.to_datetime(dates)
    dates = pd.date_range(
        start   = dap_row['startDate'], 
        end     = dap_row['endDate'], 
        periods = dap_row['Tdim']
        )

    # concatenate variable name with date and model info
    name = dap_row['variable'] + '_' + dates.strftime('%Y-%m-%d-%H-%M-%S') + '_' + dap_row['model'] + '_' + dap_row['ensemble'] + '_' + dap_row['scenario']
    name = name.str.replace('_NA', '', regex=False)

    # extract variable name
    vars = dap_row['variable']

    # if variable name is NA, use varname
    if len(vars) == 0:
        vars = dap_row['varname']

    # clean up timeseries names
    names_ts = "_".join([vars, dap_row["model"], dap_row["ensemble"], dap_row["scenario"]])
    names_ts = names_ts.replace("_NA", "")
    names_ts = names_ts.replace("__", "_")
    names_ts = names_ts.rstrip("_")

    # if dap_row has 1 column and 1 row, or 1 key/value
    if len(dap_row.keys()) == 1 and len(dap_row.values()) == 1:
        # reshape var into a 2D array
        var_2d = var.reshape((len(dates), -1))

        # create a dictionary of column names and values
        var_dict = {f'var_{i}': var_2d[:, i] for i in range(var_2d.shape[1])}
        var_dict = {key.replace("var_", f'{names_ts}_'): var_dict[key] for key in var_dict.keys()}

        # create a DataFrame with dates and var_dict as columns
        df = pd.DataFrame({'date': dates, **var_dict})

    # x resolution
    resx = (dap_row['Xn'] - dap_row['X1'])/(dap_row['ncols'] - 1)

    # y resolution
    resy = (dap_row['Yn'] - dap_row['Y1'])/(dap_row['nrows'] - 1)

    xmin = dap_row['X1'] - 0.5 * resx
    xmax = dap_row['Xn'] + 0.5 * resx
    ymin = dap_row['Y1'] - 0.5 * resy
    ymax = dap_row['Yn'] + 0.5 * resy

    # expand dimensions of 'var' if it's a 2D array
    if var.ndim == 2:
        var = np.expand_dims(var, axis=-1)

    # check if size of first dimension of 'var' is equals number of rows in 'dap'
    if var.shape[2] != dap_row["nrows"]:
        # transpose the first two dimensions of 'var' if not in correct order
        var = var.transpose(dap_row["Y_name"], dap_row["X_name"], dap_row["T_name"])
        # var2 = var.transpose(dap_row["T_name"], dap_row["X_name"],  dap_row["Y_name"])

    # Create the DataArray with the CRS included
    r = xr.DataArray(
        var,
        coords = {
            'y': -1*np.linspace(ymin, ymax, dap_row['nrows'], endpoint=False),
            'x': np.linspace(xmin, xmax, dap_row['ncols'], endpoint=False),
            'time': dates,
            'crs': dap_row['crs']
        },
        dims=['y', 'x', 'time']
        )

    # if toptobottom is True, flip the data vertically
    if dap_row['toptobottom']:

        # vertically flip each 2D array
        flipped_data = np.flip(r.values, axis=1)

        # create new xarray DataArray from flipped NumPy array
        r = xr.DataArray(
            flipped_data,
            dims   = ('y', 'x', 'time'),
            coords = {'y': r.y, 'x': r.x, 'time': r.time}
            )
        
    # set the name attribute of the DataArray
    r = r.assign_coords(time=name)

    return r

def do_dap(catalog, AOI, varname, verbose = False):
        dap_data = dap(
            catalog = catalog,
            AOI     = AOI,
            verbose = verbose
            )
        return dap_data[varname]

def repeat_durations(start_date, end_date, repeat_count):
    """Repeat durations between start_date and end_date.

    Args:
        start_date (str or datetime): Start date of the duration.
        end_date (str or datetime): End date of the duration.
        repeat_count (int): Number of times to repeat the durations.

    Returns:
        list: List of repeated durations in the format 'YYYY-MM-DD/YYYY-MM-DD'.
    """
    date_list = pd.date_range(start=start_date, end=end_date, freq='D').tolist()
    repeated_dates = []
    for date in date_list:
        for i in range(repeat_count):
            repeated_dates.append(date)
    durs = [f'{i.strftime("%Y-%m-%d")}/{i.strftime("%Y-%m-%d")}' for i in repeated_dates]
    return durs

def get_prism_daily(AOI, varname, startDate, endDate, verbose = False):

    """Retrieve PRISM daily climate data.

    Args:
        AOI (str): Area of interest.
        varname (str): Variable name.
        startDate (str or datetime): Start date of the data.
        endDate (str or datetime): End date of the data.
        verbose (bool): Verbosity flag.

    Returns:
        dict: Dictionary containing the retrieved climate data.
    """
        
    # collect raw meta data
    raw = climatepy_filter.climatepy_filter(
        id        = "prism_daily", 
        AOI       = AOI, 
        varname   = varname,
        startDate = startDate,
        endDate   = endDate
    )

    if endDate is None:
        endDate = startDate

    # collect varname in correct order
    varname = raw.varname.unique().tolist()

    # for i in range(len(varname)):
    #     # make dates for each variable
    #     pd.date_range(start=startDate, end=endDate, freq='D').tolist()
    # convert to datetime objects and create date range
    dates = pd.date_range(start=startDate, end=endDate, freq='D').tolist()

    # create new DataFrame with xx column for each date in dates
    out = [raw.assign(x=date) for date in dates]
    
    # concatenate all DataFrames into a single DataFrame
    out = pd.concat(out, ignore_index=True)

    # add YYYY column
    out = out.assign(YYYY=out['x'].apply(lambda x: x.year))

    # convert datetime column to YYYY-MM-DD string format
    out['x'] = out['x'].dt.strftime('%Y-%m-%d')

    # create a YYYYMMDD column from string date
    out = out.assign(YYYYMMDD=out['x'].apply(lambda x: x.replace("-", "")))

    # define lambda function to replace substrings in URL column
    replace_values = lambda x: x['URL'].replace('{YYYY}', str(x['YYYY'])).replace('{YYYYMMDD}', str(x['YYYYMMDD']))

    # apply lambda function to DataFrame
    out['URL'] = out.apply(replace_values, axis=1)

    # add duration column
    out['duration'] = repeat_durations(startDate, endDate, len(varname))
    # out['duration'] = [f'{i.strftime("%Y-%m-%d")}/{i.strftime("%Y-%m-%d")}' for i in dates]

    # from joblib import Parallel, delayed, parallel_backend
    out_lst = Parallel(n_jobs=-1)(delayed(do_dap) (catalog = out.iloc[[i]], 
                                                AOI  = AOI,
                                                varname = out.varname.iloc[i], 
                                                verbose = False
                                                ) for i in range(len(out)))
    # out_lst = []
    # for i in range(len(out)):
        
    #     dap_data = do_dap(
    #         catalog = out.iloc[[i]],
    #         AOI     = AOI,
    #         varname = out.varname.iloc[i],
    #         verbose = verbose
    #     )
        # out_lst.append(dap_data)
        # out_lst.append(dap_data[out.varname.iloc[i]])
    
    out_lst = merge_across_time(out_lst)

    # type(out_lst[0])
    # get data names
    out_names = list(dict.fromkeys([name.replace("_$", "") for name in out['variable'].tolist()]))

    # put list and names into a dictionary
    out = dict(zip(out_names, out_lst))

    return out

def get_nodata(dtype):
    """Returns the NoData value based on the data type.

    Args:
        dtype (str): The data type string.

    Returns:
        The NoData value corresponding to the data type. If the data type is 'int', returns 0. If the data type is 'float',
        returns np.nan. If the data type is 'uint', returns 0. If the data type is 'complex', returns 0 + 0j. If the data
        type is 'bool', returns False. For any other data type, returns None.
    """

    if "int" in dtype:
        return 0
    elif "float" in dtype:
        # return 0.0 if dtype == "float32" or dtype == "float64" else np.nan
        return np.nan
    elif "uint" in dtype:
        return 0
    elif "complex" in dtype:
        return 0 + 0j
    elif "bool" in dtype:
        return False
    else:
        return None
    

def crop_vrt2(urls, AOI, verbose = False):

    """Crop a VRT to an AOI"""

    # Function for opening a single VRT file with rasterio
    def crop_vrt_for_url(url, AOI, verbose):
        
        with rio.open(url) as src:
            if verbose: 
                print('Source tags:', src.tags(1))
                print("Source desc: ", src.descriptions)
                print("Source profile: ", src.profile)

            # Reproject the geometry to the CRS of the DataArray
            AOIv = AOI.to_crs(src.crs, inplace=False)

            # Get the bounding box of your AOI shape
            bbox = AOIv.geometry.total_bounds

            # Create a polygon object representing the bounding box
            bounding_box = shapely.geometry.box(bbox[0], bbox[1], bbox[2], bbox[3])

            if verbose:
                print(" Cropping VRT to bounding box...")

            # check data types and get nodata value
            dtype   = src.profile['dtype']
            no_data = get_nodata(dtype)

            print("dtype: ", dtype)
            print("nodata: ", no_data)

            # out_image, out_transform = rio.mask.mask(src, [bounding_box], crop=True, invert = False)
            out_image, out_transform = rio.mask.mask(src, [bounding_box], crop=True, nodata = no_data, invert = False)
            out_meta = src.meta
            out_tags = src.tags(1)

            # if nodata in out_meta, replace nodata value with value used as nodata
            if 'nodata' in out_meta:
                out_meta['nodata'] = no_data

            # if nodata in out_tags, replace nodata value with value used as nodata
            if 'nodata' in out_tags:
                out_tags['nodata'] = no_data

            # out_image, out_transform = rio.mask.mask(src, AOIv.geometry, crop=True, nodata=np.nan, invert = False)
            # out_meta = src.meta
            
            # close the dataset
            src.close()

        if out_image.ndim == 3:
            out_image = out_image.squeeze()

        # bb = AOI.geometry.total_bounds
        # rio.windows.from_bounds(*bb, out_transform)

        # get height and width of image
        height = out_image.shape[0]
        width = out_image.shape[1]

        # get x and y width height indices
        x_indices = np.arange(width)
        y_indices = np.arange(height)

        # create meshgrid
        x_coords, y_coords = np.meshgrid(x_indices, y_indices)

        # get affine transform for coordinates
        x_coords, y_coords = rio.transform.xy(out_transform, y_coords, x_coords)

        # stack the arrays along a new dimension
        coords = np.stack((x_coords, y_coords), axis=-1)

        # x and y stacks
        xn = np.stack((x_coords), axis=-1)
        yn = np.stack((y_coords), axis=-1)

        # get min and max of x and y
        xmin = xn.min()
        xmax = xn.max()
        ymin = yn.min()
        ymax = yn.max()

        # # GET CRS
        # crs = out_meta['crs']

        # create DataArray
        r = xr.DataArray(
            out_image,
            coords={
                'y': -1*np.linspace(ymin, ymax, height, endpoint=False),
                'x': np.linspace(xmin, xmax, width, endpoint=False),
            },
            dims=['y', 'x']
        )

        # add tags to attributes of data array
        for key, value in out_tags.items():
            r.attrs[key] = value

        # add tags to attributes of data array
        for key, value in out_meta.items():
            r.attrs[key] = value

        return r
    

    # tmp = crop_vrt_for_url(urls[1], AOI, verbose = True)
    results = Parallel(n_jobs=-1)(delayed(crop_vrt_for_url) (url = i,
                                            AOI  = AOI,
                                            verbose = False
                                            ) for i in urls)
    
    return results

def crop_vrt(urls, AOI, verbose = False):

    """Crop a VRT to an AOI"""

    # make empty list to store dataarrays
    da_lst = []

    # loop over each url
    for url in urls:
        with rio.open(url) as src:
            if verbose: 
                print('Source tags:', src.tags(1))
                print("Source desc: ", src.descriptions)
                print("Source profile: ", src.profile)

            # Reproject the geometry to the CRS of the DataArray
            AOIv = AOI.to_crs(src.crs, inplace=False)

            # Get the bounding box of your AOI shape
            bbox = AOIv.geometry.total_bounds

            # Create a polygon object representing the bounding box
            bounding_box = shapely.geometry.box(bbox[0], bbox[1], bbox[2], bbox[3])

            if verbose:
                print(" Cropping VRT to bounding box...")

            # check data types and get nodata value
            dtype   = src.profile['dtype']
            no_data = get_nodata(dtype)

            print("dtype: ", dtype)
            print("nodata: ", no_data)

            # out_image, out_transform = rio.mask.mask(src, [bounding_box], crop=True, invert = False)
            out_image, out_transform = rio.mask.mask(src, [bounding_box], crop=True, nodata = no_data, invert = False)
            out_meta = src.meta
            out_tags = src.tags(1)

            # if nodata in out_meta, replace nodata value with value used as nodata
            if 'nodata' in out_meta:
                out_meta['nodata'] = no_data

            # if nodata in out_tags, replace nodata value with value used as nodata
            if 'nodata' in out_tags:
                out_tags['nodata'] = no_data

            # out_image, out_transform = rio.mask.mask(src, AOIv.geometry, crop=True, nodata=np.nan, invert = False)
            # out_meta = src.meta
            
            # close the dataset
            src.close()

        if out_image.ndim == 3:
            out_image = out_image.squeeze()

        # bb = AOI.geometry.total_bounds
        # rio.windows.from_bounds(*bb, out_transform)

        # get height and width of image
        height = out_image.shape[0]
        width = out_image.shape[1]

        # get x and y width height indices
        x_indices = np.arange(width)
        y_indices = np.arange(height)

        # create meshgrid
        x_coords, y_coords = np.meshgrid(x_indices, y_indices)

        # get affine transform for coordinates
        x_coords, y_coords = rio.transform.xy(out_transform, y_coords, x_coords)

        # stack the arrays along a new dimension
        coords = np.stack((x_coords, y_coords), axis=-1)

        # x and y stacks
        xn = np.stack((x_coords), axis=-1)
        yn = np.stack((y_coords), axis=-1)

        # get min and max of x and y
        xmin = xn.min()
        xmax = xn.max()
        ymin = yn.min()
        ymax = yn.max()

        # # GET CRS
        # crs = out_meta['crs']

        # create DataArray
        r = xr.DataArray(
            out_image,
            coords={
                'y': -1*np.linspace(ymin, ymax, height, endpoint=False),
                'x': np.linspace(xmin, xmax, width, endpoint=False),
            },
            dims=['y', 'x']
        )

        # add tags to attributes of data array
        for key, value in out_tags.items():
            r.attrs[key] = value

        # add tags to attributes of data array
        for key, value in out_meta.items():
            r.attrs[key] = value
        # [r.attrs.update({key: value}) for key, value in out_tags.items()]    

        da_lst.append(r)
        # dataarrays[url] = r

    return da_lst

def vrt_crop_get(
        URL         = None, 
        catalog     = None, 
        AOI         = None, 
        grid        = None,
        varname     = None, 
        start       = None, 
        end         = None, 
        toptobottom = False, 
        verbose     = False
        ):
    
#     """
#     Crop and process VRT data.

#     Args:
#         URL (str or list, optional): The URL(s) of the VRT file(s) to open. If not provided, it is extracted from the catalog.
#         catalog (object, optional): The catalog object containing the URL(s) of the VRT file(s). Required if URL is not provided.
#         AOI (geopandas.GeoDataFrame, optional): The Area of Interest polygon to crop the VRT data to.
#         grid (object, optional): The grid object defining the extent and CRS for cropping and reprojection.
#         varname (str, optional): The name of the variable to select from the VRT data.
#         start (int, optional): The start index for subsetting bands in the VRT data.
#         end (int, optional): The end index for subsetting bands in the VRT data.
#         toptobottom (bool, optional): Whether to flip the data vertically.
#         verbose (bool, optional): Whether to print informative messages during processing. Default is False

#     Returns:
#         xr.DataArray: The cropped and processed VRT data.

#     """

    if URL is None:
        URL = catalog.URL.to_list()

    if verbose:
        print("Opening VRT from URL: ", URL)

    # Area of interest
    vrts = crop_vrt(urls = URL, AOI = AOI, verbose = verbose)

    # vrts2 = Parallel(n_jobs=-1)(delayed(crop_vrt) (urls = [i],
    #                                     AOI  = AOI,
    #                                     verbose = False
    #                                     ) for i in URL)

    # check if data needs to be vertically flipped
    for idx, val in enumerate(catalog['toptobottom']):

        if verbose:
            print("idx:", idx, "val: ",val)

        if val and not np.isnan(val):

            if verbose:
                print("Flipping data vertically")
            # vertically flip each 2D array
            flipped_data = np.flip(vrts[idx].values, axis=0)

            # stash tags
            tags = vrts[idx].attrs

            # create new xarray DataArray from flipped NumPy array
            vrts[idx] = xr.DataArray(
                flipped_data,
                dims   = ('y', 'x'),
                coords = {'y': vrts[idx].y, 'x': vrts[idx].x}
                )
            
            # add tags back to flipped DataArray
            vrts[idx].attrs = tags

        else:
            if verbose:
                print("Not flipping data vertically")
            # vrts[idx] = np.flip(vrts[idx], axis=0)

    # create dictionary of DataArrays
    vrts = dict(zip(catalog['variable'], vrts))

    return vrts

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

# def vrt_crop_get2(
#         URL         = None, 
#         catalog     = None, 
#         AOI         = None, 
#         grid        = None,
#         varname     = None, 
#         start       = None, 
#         end         = None, 
#         toptobottom = False, 
#         verbose     = False
#         ):
    
#     """
#     Crop and process VRT data.

#     Args:
#         URL (str or list, optional): The URL(s) of the VRT file(s) to open. If not provided, it is extracted from the catalog.
#         catalog (object, optional): The catalog object containing the URL(s) of the VRT file(s). Required if URL is not provided.
#         AOI (geopandas.GeoDataFrame, optional): The Area of Interest polygon to crop the VRT data to.
#         grid (object, optional): The grid object defining the extent and CRS for cropping and reprojection.
#         varname (str, optional): The name of the variable to select from the VRT data.
#         start (int, optional): The start index for subsetting bands in the VRT data.
#         end (int, optional): The end index for subsetting bands in the VRT data.
#         toptobottom (bool, optional): Whether to flip the data vertically.
#         verbose (bool, optional): Whether to print informative messages during processing.

#     Returns:
#         xr.DataArray: The cropped and processed VRT data.

#     """

#     if URL is None:
#         URL = catalog.URL.to_list()

#     if verbose:
#         print("Opening VRT from URL: ", URL)

#     # Read in each file as a separate DataArray and put them in a list
#     vrts = [xr.open_rasterio(url) for url in URL]

#     # Concatenate the DataArrays along the band dimension

#     if len(vrts) == 1:
#         vrts = vrts[0]
#     else:
#         vrts = xr.concat(vrts, dim="band")
    
#     # open VRT
#     # with xr.set_options(keep_attrs=True):
#     # vrts = rxr.open_rasterio(URL[0])

#     # index number and name of index for subsetting bands
#     # var_idx = vrts.attrs['long_name'].index(varname.item()) 
#     # var_key = [i for i in vrts.attrs['long_name'] if i == varname.item()]

#     # if varname is Not none
#     if varname is not None:
        
#         # vrts = vrts.isel(band=vrts.attrs['long_name'].index(varname.item()))
#         if "long_name" in vrts.attrs.keys():

#             if verbose:
#                 print("Selecting varnames")
#             vrts = vrts.sel(band = vrts.attrs['long_name'].index(varname.tolist()[0]))

#     # subset by index if non "date" dimension
#     if start is not None and end is None:
#         vrts = vrts.isel(band=start)
#     elif start is not None and end is not None:
#         vrts = vrts.isel(band=slice(start, end))

#     if grid is not None:
#         xmin = grid.extent[0]
#         ymin = grid.extent[1]
#         xmax = grid.extent[2]
#         ymax = grid.extent[3]

#         # clip vrts to grid extent
#         vrts = vrts.rio.clip_box(xmin, ymin, xmax, ymax)
        
#         # reproject vrts to grid CRS
#         vrts.rio.write_crs(grid.crs, inplace=True)

#         # flag as True if grid is given
#         flag = True
#     else:
#         if (vrts.rio.crs.to_string() is None or vrts.rio.crs.to_string() == "") or all([i in [0, 1, 0, 1] for i in vrts.rio.bounds()]):
#             if verbose:
#                 print("Defined URL(s) are aspatial and on a unit grid. Cannot be cropped")

#             # flag as False if missing CRS or if bounding box is [0, 1, 0, 1]
#             flag = False
#         else:
#             # flag as True if no grid is given and nothing noteworthy
#             flag = True

#     # if an AOI is given and no flagging happens, crop and mask rasters to AOI
#     if AOI is not None and flag:

#         # Reproject the geometry to the CRS of the DataArray
#         AOIv = AOI.to_crs(vrts.rio.crs)
    
#         # reproject AOI to vrts CRS
#         # AOIv = AOI.to_crs(vrts.rio.crs, inplace=False).geometry.apply(lambda x: (x, 1))

#         if verbose:
#             print("Cropping/Clipping to AOI")
#         # vrts.rio.
#         # Crop the raster to the extent of the AOI
#         # Crop and mask the DataArray to the polygon
#         vrts = vrts.rio.clip(AOIv.geometry.apply(mapping))

#         # dim that is NOT x or y
#         selected_dim = [dim for dim in vrts.dims if dim not in ['x', 'y']][0]

#         if "long_name" in vrts.attrs.keys():
#             # delete extra long_name attributes
#             del vrts.attrs['long_name']
    
#     # if toptobottom is True, flip the data vertically
#     if toptobottom:
        
#         # print("Flipping data toptobottom")
#         if verbose:
#             print("Flipping data vertically")

#         # vertically flip each 2D array
#         flipped_data = np.flip(vrts.values, axis=0)

#         # create new xarray DataArray from flipped NumPy array
#         vrts = xr.DataArray(
#             flipped_data,
#             dims   = ('band', 'y', 'x'),
#             coords = {'band': vrts[selected_dim], 'y': vrts.y, 'x': vrts.x}
#             # dims   = ('y', 'x', 'time'),
#             # coords = {'y': vrts_crop.y, 'x': vrts_crop.x, 'time': vrts_crop.time}
#             )
#     # vrts.close()

#     return vrts