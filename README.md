# **climatePy**

Provide a short description explaining the what, why, and how of your project. Use the following questions as a guide:

<!-- badges: start -->

[![stage](https://img.shields.io/badge/stage-dev-orange)](#)
[![Dependencies](https://img.shields.io/badge/dependencies-04/12-orange?style=flat)](#)
[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://choosealicense.com/licenses/mit/)

<!-- badges: end -->

## Description

A Python 📦 for getting point and gridded climate data by AOI. `climatePy` is the Python version of the [`climateR`](https://github.com/mikejohnson51/climateR) R package, providing all of the same functionality but in Python.

As its stated in the [climateR README](https://github.com/mikejohnson51/climateR#climater):
climatePy simplifies the steps needed to get climate data into Python. At its core it provides three main things:

1. A climate catalog of over 100,000k datasets from over 2,000 data providers/archives. See (`load_data()`)

2. A general toolkit for accessing remote and local gridded data files bounded by space, time, and variable constraints (`dap()`, `dap_crop()`, `read_dap_file()`)

3. A set of shortcuts that implement these methods for a core set of selected catalog elements

<br>

---

- A slideshow walking through the capabilities of [**climateR/climatePy**](https://mikejohnson51.github.io/climateR-intro/#1)

---

## Table of Contents (Optional)

- [Installation](#installation)
- [Usage](#usage)
- [Credits](#credits)
- [License](#license)
- [How to Contribute](#how-to-contribute)

<br>

## Installation

`climatePy` is still in the development phase and is not yet ready for external users, but eventually it will be installable from PyPI like so:

``` 
pip install climatePy
```

<br>

## Usage

### Loading climate catalog

```python
import geopandas as gpd
import matplotlib.pyplot as plt
from src.climatePy import load_data

# load climate catalog
catalog = load_data()

# load example AOI data
AOI = gpd.read_file('src/data/san_luis_obispo_county.gpkg')
```
<br>

### Using `climatepy_filter()`:

The `climatepy_filter()` is one of the core functions of `climatePy` and is used to do the first round of filtering on the base climate catalog.

Here we filter down our climate catalog to TerraClim precipitation data for San Luis Obispo County, CA.

```python
# collect raw meta data
raw = climatepy_filter.climatepy_filter(
        id        = "terraclim", 
        AOI       = AOI, 
        varname   = "ppt"
        )
```

| id  | asset | varname    |
|-------|-----|---------|
| gridmet | agg_terraclimate_ppt_1958_CurrentYear_GLOBE  | ppt   |

### AOI
![San Luis Obispo County county](assets/images/san_luis_obispo_county_polygon.png)

<br>

### Getting climate data in AOI

Now lets use the `getTerraClim()` function from `climatePy` to get precipitation data for January 1st, 2018 in San Luis Obispo County, CA.

```python
# collect raw meta data
prcp = shortcuts.getTerraClim(
    AOI       = AOI,
    varname   = "ppt",
    startDate = "2018-01-01",
    endDate   = "2018-01-01"
    )
```
![Precipitation San Luis Obispo County](assets/images/san_luis_obispo_county_ppt.png)

<br>
<br>

## Credits

Credit to [Mike J Johnson](https://github.com/mikejohnson51) and the other contributors to the original [`climateR`](https://github.com/mikejohnson51/climateR) package listed below:
- [Max Joseph](https://github.com/mbjoseph)
- [Eric R. Scott](https://github.com/Aariq)
- [James Tsakalos](https://github.com/jamestsakalos)

<br>

## License

MIT License

Copyright (c) 2023 Angus Watters, Mike J. Johnson

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

---

<br>

## How to Contribute
If you would like to contribute, submit a PR and we will get to as soon as we can!
If you have any issues please open an issue on GitHub. For any questions, feel free to ask [@anguswg-ucsb](https://github.com/anguswg-ucsb) or [@mikejohnson51](https://github.com/mikejohnson51), or simply create an issue on GitHub.