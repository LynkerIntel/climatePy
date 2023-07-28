import re
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd

def clean_time(df, col, inplace=False):
    # regular expression to extract numbers between dashes (date strings)
    regex_pat = r'(\d+)-'

    # find all regular expression matches
    clean_time = ["-".join(re.findall(regex_pat, value)) for value in df[col]]

    def correct_date(input_string):
        match = re.match(r'(\d{4}-\d{2}-\d{2})-(.*)', input_string)

        if match:
            date_part = match.group(1)
            time_part = match.group(2)

            formatted_time = re.sub(r'[^a-zA-Z0-9]', ':', time_part)

            return date_part + 'T' + formatted_time

        else:
            raise ValueError("Invalid input string format.")
    if inplace:
        df['date'] = [np.datetime64(correct_date(input_string)) for input_string in clean_time]
        return df
    else:
        return [np.datetime64(correct_date(input_string)) for input_string in clean_time]
    
def pts_extracter(r, pts, id=None):
    """
    Extracts data from an xarray DataArray at specific points based on a GeoDataFrame of points.

    Args:
        r (xarray.DataArray): The xarray DataArray object containing the data.
        pts (geopandas.GeoDataFrame): The GeoDataFrame containing the points of interest.
        id (str, optional): The column name in 'pts' GeoDataFrame to use as the identifier.
            If not provided, it uses the index values as the identifier. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame with extracted data for each point and time.

    Raises:
        TypeError: If 'r' is not an xarray DataArray object.
        ValueError: If 'pts' GeoDataFrame does not have a CRS attribute or if the 'id' column is not found in 'pts'.
    """

    if not isinstance(r, xr.DataArray):
        raise TypeError("'r' must be an xarray DataArray object.")
    
    if pts.crs is None:
            raise ValueError("'pts' GeoDataFrame does not have a CRS attribute. Please set a CRS attribute for the 'pts' GeoDataFrame.")
    

    if id is not None and id not in pts.columns:
        raise ValueError(f"id column '{id}' not found in pts GeoDataFrame.")
    
    if id is None:
        pts['uid'] = pts.index
        id = 'uid'
    
    # make a hashmap of names and index values
    names_map = {i: v[0] for i, v in enumerate(zip(pts[id], pts.index))}

    # transform point to CRS of xarray
    pts = pts.to_crs(r['crs'].values.tolist()) 
    
    # get X and Y coordinates of points as data arrays
    target_x = xr.DataArray(pts.geometry.x, dims=id)
    target_y = xr.DataArray(pts.geometry.y, dims=id)
    
    pts_data = r.sel(x=target_x, y=target_y, method="nearest")

    pts_df = pts_data.to_dataframe().reset_index()
    
    # # preserve original data variable name
    varname = r.name

    # select columns of interest
    pts_df = pts_df[['time', varname, id]]
    
    # replace the dataframe index with the hashmap values
    pts_df[id] = pts_df[id].map(names_map)

    # pivot data wide
    pts_df = pts_df.pivot(index='time', columns=id, values=varname)

    # reset index
    pts_df = pts_df.reset_index()
    
    # # extract varname from time column
    # pts_df['varname'] = clean_varname(pts_df, "time", inplace=False)
    # # pts_df['varname'] = varname

    # convert time column to datetime
    pts_df['time'] = clean_time(df = pts_df, col = "time", inplace=False)
    # pts_df['time'] = clean_time(pts_df, "time")

    # extract varname from time column
    pts_df['varname'] = varname
    # pts_df['varname'] = clean_varname(pts_df, "time", inplace=False)

    # rename "time" column to "date"
    pts_df.rename(columns={'time': 'date'}, inplace=True)

    # reorder columns putting date and varname first
    pts_df = pts_df[['date', 'varname'] + [x for x in pts_df.columns if x not in ['date', 'varname']]]

    return pts_df

def extract_sites(r, pts, id=None):
    """
    Extracts data from xarray DataArray(s) at specific points based on a GeoDataFrame of points.

    Args:
        r (xarray.DataArray or dict): The xarray DataArray object(s) or dictionary of DataArrays.
        pts (geopandas.GeoDataFrame): The GeoDataFrame containing the points of interest.
        id (str, optional): The column name in 'pts' GeoDataFrame to use as the identifier.
            If not provided, it uses the index values as the identifier. Defaults to None.

    Returns:
        pandas.DataFrame: DataFrame with extracted data for each point and time.

    Raises:
        TypeError: If 'r' is not an xarray DataArray object or a dictionary.
    """

    # if input is a single data array
    if isinstance(r, xr.DataArray):
        res = pts_extracter(r=r, pts=pts, id=id)

        return res

    # if input is a dictionary of data arrays
    if isinstance(r, dict):

        res = []

        for key, value in r.items():
            # print(f"key: {key}")
            # print(f"EXTRACTING POINTS FOR VARIABLE: {key}")

            res.append(pts_extracter(r=value, pts=pts, id=id))

            # print(f'---------')

        res = pd.concat(res, ignore_index=True)

    return res

# def clean_time(df, col, inplace=False):

#     regex = r".*?(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})"

#     clean_time = [re.match(regex, value).group(1) for value in df[col]]

#     def correct_date(input_string):
#         match = re.match(r'(\d{4}-\d{2}-\d{2})-(.*)', input_string)

#         if match:
#             date_part = match.group(1)
#             time_part = match.group(2)

#             formatted_time = re.sub(r'[^a-zA-Z0-9]', ':', time_part)

#             return date_part + 'T' + formatted_time

#         else:
#             raise ValueError("Invalid input string format.")
#     if inplace:
#         df['date'] = [np.datetime64(correct_date(input_string)) for input_string in clean_time]
#         return df
#     else:
#         return [np.datetime64(correct_date(input_string)) for input_string in clean_time]
    
# def clean_varname(df, col, inplace=False):

#     # regular expression
#     regex = r'^(.*?)\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}$'

#     # extract text BEFORE date
#     extracted_text = [re.match(regex, value).group(1) for value in df[col]]

#     # remove trailing underscores
#     extracted_text = [s.rstrip('_') for s in extracted_text]

#     if inplace:
#         df['varname'] = extracted_text
#         return df
#     else:
#         return extracted_text
    