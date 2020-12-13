
import pandas as pd
import numpy as np
import os
import datetime

'''
First copy the file Main_Query.csv on the local
repository file
'''
# Load the data and create dataframes
Dataframe = pd.read_csv(r'Main_Query.csv')
Dataframe.head()
Dataframe.isna().mean()
# Droping rows with defective data on user_id and amount payed
PaymentNa = Dataframe[Dataframe['attr_price_num'].isnull()]
Dataframe = Dataframe.dropna(subset=['user_id', 'attr_price_num'])
Dataframe = Dataframe.drop(columns='event_action')

# Convert logtime to date format
Dataframe["logtime"] = pd.to_datetime(Dataframe["logtime"])
Dataframe.groupby('nc')['attr_price_num'].mean()
df = Dataframe.loc[Dataframe['attr_label_str'] != 'loyalty']
monte = df.loc[df['nc'] == 'me']
monte.describe()
df.isna().mean()
df.groupby('nc')['attr_price_num'].mean()
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

# Function that convert local currency to EUR


class currencyConverter:

    def __init__(self, df):
        self.df = df
        self.NatCoNames = ['at', 'cz', 'heyah', 'hr', 'hu', 'me', 'mk', 'pl',
                           'ro', 'sk']
        self.rates = [1, 0.038, 0.22, 0.13, 0.0028, 1, 0.016, 0.22, 0.21, 1]

    def convert(self, nc, amount):
        rate = self.rates[self.NatCoNames.index(nc)]
        return amount*rate

    def ConvertColumn(self):
        self.df['price_eur'] = self.df.apply(lambda x:
                                                      self.convert(x['nc'],
                                                                   x[
                                                               'attr_price_num'
                                                               ]), axis=1)
        return self.df


# Function that calculates average return per user
def Return_perUser(dataframe, date_today):
    date_today = pd.to_datetime(date_today)
    unique_users = pd.Series(dataframe['user_id'].unique())
    average_revenue = []
    for user in unique_users:

        subsection = dataframe.loc[dataframe['user_id'] == user]
        revenue_of_user = subsection['price_eur']
        oldest_date = min(subsection['logtime'])
        months = GetMonths(oldest_date, date_today)
        total_revenue = revenue_of_user.sum()
        avg_monthly_revenue = Division(total_revenue, months)
        average_revenue.append(avg_monthly_revenue)
    dataframe_converted = unique_users.to_frame(name='user_id')
    dataframe_converted['avg_revenue'] = average_revenue
    return dataframe_converted


# First we create a column with all the payments in euros
currencyConverter_01 = currencyConverter(df)
df_euros = currencyConverter_01.ConvertColumn()
df_euros = df_euros.sort_values(by=['user_id'])
df_euros.groupby('nc')['price_eur'].mean()
# We create a dataframe with avg monthly revenue per user
avg_monthly_revenue = Return_perUser(df_euros,
                                     '2020-11-30 20:29:14.169000+00:00')

avg_monthly_revenue_merged = avg_monthly_revenue.merge(df_euros[['user_id',
                                                                 'nc',
                                                                 'attr_os_str',
                                                                 'devicebrand',
                                                                 'cnt_call',
                                                                 'cnt_dis']],
                                                        on='user_id',
                                                        how='inner')
avg_monthly_revenue_merged.isna().mean()
df1 = avg_monthly_revenue_merged.drop_duplicates(keep='first',
                                                 subset='user_id')
df1.isna().mean()
# Load dataframe with services and add ons
df2 = pd.read_csv(r'services_and_add-ons.csv')
df2.head()
df2.isna().mean()
df2 = df2.dropna(subset=['user_id'])
# Merge both dataframes
df_merged = df1.merge(df2, on='user_id', how='left')
df_merged.isna().mean()
df_merged.describe()
# Drop users without any services
df_merged = df_merged.dropna(subset=['cnt_mobile', 'cnt_internet', 'cnt_tv',
                                     'cnt_voice'], how='all')
df_merged.describe()
# Replace NA with 0
df_merged = df_merged.fillna(0)
# Print datafame
df_merged.to_csv('df_merged.csv', index=False)
