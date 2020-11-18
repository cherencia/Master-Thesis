import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime as date

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

Dataframe_02.groupby('user_id')['attr_price_num'].mean()

# Define function to get months between two dates


def GetMonths(t0, date_today):
    delta = date_today - t0
    months = int(delta.days/30)
    return months
# Function that calculates average return per user


def Return_perUser(dataframe, date_today):
    unique_users = pd.Series(dataframe['user_id'].unique())
    for user in unique_users:
        subsection = dataframe.loc[dataframe['user_id'] == user]
        oldest_date = min(subsection['logtime'])
        months = GetMonths(oldest_date, date_today)
        avg_monthly_revenue =
