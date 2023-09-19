import setuptools
# from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="climatePy",                         # pkg name
    # version="0.0.4.24",                        # version
    version='{{VERSION_PLACEHOLDER}}',
    author="Angus Watters, Mike Johnson",     # authors
    author_email = "anguswatters@gmail.com, mikecp11@gmail.com",
    description="A Python package for getting point and gridded climate data by AOI",
    long_description=long_description,      # long description is read from the the readme file
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),    # python modules to install
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],          
    python_requires='>=3.6',               # min python version required 
    # py_modules=["climatePy"],             #  python package name
    # package_dir={'':'climatePy/src'},     #  source code directory of package
    # package_dir={'':'src/climatePy'},     #  source code directory of package
    # package_dir={'':'/climatePy'},     #  source code directory of package
    install_requires=['pandas', 'pyarrow', 'geopandas', 'shapely', 'pyproj', 'rasterio', 
                    'xarray', 'rioxarray', 'rtree', 
                    'numpy', 'geogif', 'netCDF4', 'joblib'], # dependencies
    include_package_data=True, 
    # package_data={'': ['data/*.csv']}     # include catalog csv dataset
    package_data={'': ['data/*.parquet']}     # include catalog parquet dataset
    # package_data={'': ['data/*.csv', 'data/*.parquet']}     # include catalog csv AND parquet dataset
    # package_data={'src/climatePy': ['src/data/*']}
    )