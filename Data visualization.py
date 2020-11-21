import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


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
Dataframe_02 = Dataframe_02.dropna(subset=['attr_price_num', 'logtime'])

Dataframe_02.groupby('user_id')['attr_price_num'].mean()

# Define function to get months between two dates


def GetMonths(t0, date_today):
    delta = date_today - t0
    months = int(delta.days/30)
    return months


# Create function that eludes dividing by zero
def Division(n, d):
    return n / d if d else 0


# Function that calculates average return per user
def Return_perUser(dataframe, date_today):
    date_today = pd.to_datetime(date_today)
    unique_users = pd.Series(dataframe['user_id'].unique())
    average_revenue = []
    for user in unique_users:
        subsection = dataframe.loc[dataframe['user_id'] == user]
        revenue_of_user = subsection['attr_price_num']
        oldest_date = min(subsection['logtime'])
        months = GetMonths(oldest_date, date_today)
        total_revenue = revenue_of_user.sum(axis=0)
        avg_monthly_revenue = Division(total_revenue, months)
        average_revenue.append(avg_monthly_revenue)
    dataframe_converted = unique_users.to_frame(name='user_id')
    dataframe_converted['avg_revenue'] = average_revenue
    return dataframe_converted


# Now we have a dataframe with the monthly return on average
avg_monthly_revenue = Return_perUser(Dataframe_02, '2020-11-21')


# Merge datasets
