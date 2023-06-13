import pandas as pd
import pkg_resources

def load_data():
    data_file = pkg_resources.resource_filename('src', 'data/catalog.csv')
    data = pd.read_csv(data_file)
    return data