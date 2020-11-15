import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime

# Load the data and create dataframes
os.chdir(r'C:\Users\Carlosh\github\Master-Thesis')
Dataframe_01 = pd.read_csv(
           r'C:\Users\Carlosh\github\Master-Thesis\Dataframe_01.csv')
Dataframe_02 = pd.read_csv(
           r'C:\Users\Carlosh\github\Master-Thesis\Dataframe_02.csv')

Dataframe_01.head()
Dataframe_02.head()

# new data frame with split value columns

Dataframe_02["logtime"] = Dataframe_02["logtime"].str.split("T",
                                                            n=1, expand=True)
Dataframe_02["logtime"] = pd.to_datetime(Dataframe_02["logtime"])
