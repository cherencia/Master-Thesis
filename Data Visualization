import matplotlib.pyplot as plt
import pandas as pd
from pandas_profiling import ProfileReport

# Load dataframe
df_merged = pd.read_csv(r'df_merged.csv')
prof = ProfileReport(df_merged)
prof.to_file(output_file='report.html')
# IOS or android
ios_or_android = df_merged.groupby('attr_os_str')['attr_os_str'].count()
ios_or_android.plot.bar()
# NatcCos
natco = df_merged.groupby('nc')['nc'].count()
natco.plot.bar()
# ARPU
ARPU = df_merged.groupby('nc')['avg_revenue'].mean()
ARPU.plot.bar()
