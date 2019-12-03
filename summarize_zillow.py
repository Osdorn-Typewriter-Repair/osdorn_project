import numpy as np
import pandas as pd
from env import host, user, password
import acquire_z

df = acquire_z.acquire_zillow()

def summarize(df):
    df.head()
    df.tail(5)
    df.sample(5)
    df.describe()
    df.shape
    df.isnull().sum()
    df.info()
    return df
summarize_z = summarize(df)