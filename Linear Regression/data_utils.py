from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_data(filepath="data/housing.csv"):
    df = pd.read_csv(filepath)
    return df
   

housing = load_data()
print(housing.info())
print(housing.head())