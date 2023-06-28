import setuptools
# from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="climatePy",                         # pkg name
    version="0.0.3.0",                        # version
    author="Angus Watters",                     # author
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
    install_requires=['pandas', 'geopandas', 'shapely', 'pyproj', 'rasterio', 'xarray', 'rtree', 'numpy',
                    'netCDF4', 'joblib'], # dependencies
    include_package_data=True, 
    package_data={'': ['data/*.csv']}     # include catalog csv dataset
    # package_data={'src/climatePy': ['src/data/*']}
)