import geogif
import os 
import xarray as xr

def animiation_xarray(data, outfile, color):
    """Create an animation GIF from an xarray DataArray.

    Parameters:
        data (xarray.core.dataarray.DataArray): The xarray DataArray to create the animation from.
        outfile (str): The output file path for the GIF.
        color (str): The colormap to use for the GIF. Can be a matplotlib colormap name as a string, or a Colormap object for custom colormapping.
        If None (default), the default matplotlib colormap (usually viridis) will automatically be used for 1-band data. 

    Returns:
        str: The output file path for the GIF.

    Raises:
        ValueError: If one of the required bands ('x', 'y', 'time') is missing in the xarray DataArray.

    Note:
        This function assumes the 'data' dimensions are in the order: ('time', 'y', 'x').
        The output GIF is created using geogif.gif().
    """

    # if one of the required bands is missing, raise a ValueError
    if not all(i in ["x", "y", "time"] for i in data.dims):
        raise ValueError("Missing xarray band, xarray must contain bands 'time', 'y', and 'x'")
    
    # make sure data is in the right order for geogif.gif() function 
    data = data.transpose('time', 'y', 'x')

    # create gif and save out
    geogif.gif(
        arr         = data,
        date_format = False, 
        cmap        = color, 
        to          = outfile
        )
    
    return outfile

def animiation_raster(data, outfile, colors = None):
    
    """Create animation GIFs from a dictionary of xarray DataArrays or a single xarray DataArray.

    Parameters:
        data (xarray.core.dataarray.DataArray, dict): The xarray DataArray(s) to create the animations from.
            If a dictionary, outfile must either be a string referencing a directory to save all of the DataArray gifs too or a list of file paths all ending with ".gif" that is the same length as the data dict object.
            If a base directory string is given, the dict keys are used for naming the GIF files.
        outfile (Union[str, List[str]]): The output file path(s) for the GIF(s).
            If a string ending in '.gif', the same output file will be used for all animations with a number suffix.
            If a list of '.gif' files, each file will be used for a corresponding key in the data dictionary.
            If a base directory string is given, the dict keys are used for naming the GIF files.
        colors (str): Colormap to use for single-band data. Can be a matplotlib colormap name as a string, or a Colormap object for custom colormapping.
        If None (default), the default matplotlib colormap (usually viridis) will automatically be used for 1-band data. 

    Returns:
        (str or List[str]): A single string indicating where filepath the GIF was
        saved to or a List of output file paths indicating where the GIF for each one of the dictionary keys was saved to
    """
    
    # if data is a DataArray, make the gif and return the file path name
    if isinstance(data, (xr.core.dataarray.DataArray, xr.core.dataarray.DataArray)):
        animiation_xarray(data = data, outfile = outfile, colors = colors)
        
        return outfile
    
    # if outfile is a string
    if isinstance(outfile, str):
        # If outfile is a string ending in .gif, use it for all data keys
        if outfile.lower().endswith('.gif'):
            outfile = [f"{os.path.splitext(outfile)[0]}_{i+1}.gif" for i in range(len(data))]
        else:
            # base directory of outfile
            out_dir = os.path.dirname(outfile)
            
            # Construct new file paths based on the out_dir and data keys
            outfile = [os.path.join(out_dir, f"{key}.gif") for key in data.keys()]

    # if the outfile list is NOT long enough, just make outfile paths using the base directory of the 
    if len(outfile) != len(data):

        # base directory of first element in outfile
        out_dir = os.path.dirname(outfile[0])

        # Construct new file paths based on the out_dir and data keys
        outfile = [os.path.join(out_dir, f"{key}.gif") for key in data.keys()]

    if isinstance(data, dict):

        for i, key in enumerate(data):
            animiation_xarray(data = data[key], outfile = outfile[i], colors = colors)
        
    return outfile