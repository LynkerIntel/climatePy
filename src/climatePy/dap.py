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
