import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import datetime

# Load the data and create dataframes


Dataframe = pd.read_csv(
          r'Final_dataframe.csv')
Dataframe.head()
Dataframe["logtime"] = pd.to_datetime(Dataframe["logtime"])
Dataframe.groupby('nc')['attr_price_num'].mean()
# Drop e_commerce purchase values (they are repeated)


df = Dataframe[Dataframe['event_action']=='Payment Completed']
df = df.loc[Dataframe['attr_label_str'] != 'loyalty']


# Define function to get months between two dates


def GetMonths(t0, date_today):
    delta = date_today - t0
    months = int(delta.days/30)
    if months != 0:
        return months
    else:
        return 1


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
        total_revenue = revenue_of_user.sum()
        avg_monthly_revenue = Division(total_revenue, months)
        average_revenue.append(avg_monthly_revenue)
    dataframe_converted = unique_users.to_frame(name='user_id')
    dataframe_converted['avg_revenue'] = average_revenue
    return dataframe_converted


# Now we have a dataframe with the monthly return on average
avg_monthly_revenue = Return_perUser(df, '2020-11-21 20:29:14.169000+00:00')
avg_monthly_revenue_merged = avg_monthly_revenue.merge(df[['user_id',
                                                                  'nc',
                                                                  'devicebrand',
                                                                  'attr_os_str']],
                                                on='user_id',how= 'inner')
avg_monthly_revenue_unique = avg_monthly_revenue_merged.drop_duplicates(keep=
                                                                        'first',
                                                                        subset=
                                                                        'user_id')
avg_monthly_revenue_unique.isna().mean()

# Merge both dataframes
Dataframe_final = df.merge(avg_monthly_revenue, on='user_id')



#Plots
ios_or_android = avg_monthly_revenue_unique.groupby(
    'attr_os_str')['attr_os_str'].count()
ios_or_android.plot.bar()
natco = avg_monthly_revenue_unique.groupby('nc')['nc'].count()
natco.plot.bar()
ARPU = avg_monthly_revenue_unique.groupby('nc')['avg_revenue'].mean()
ARPU.plot.bar()
print(os.getcwd())
