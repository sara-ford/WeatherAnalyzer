import pandas as pd
import numpy as np

df = pd.read_csv('weather_data.csv')

print(df.head())

df['Date'] = pd.to_datetime(df['Date'])

def monthly_averages(df):

    df['month']=df['Date'].dt.month

    monthly_avg = {}

    for month in range(1,13):
        month_data =df[df['month']==month]

        avg_temp=np.mean(month_data['Temperature'])
        avg_humidity=np.mean(month_data['Humidity'])
        avg_rain = np.mean(month_data['Rain'])

        monthly_avg[month] = {
         'Avg_Temperature':avg_temp,
         'Avg_Humidity':avg_humidity,
         'Avg_Rain':avg_rain
      }
    return monthly_avg
results = monthly_averages(df)

for month, stats in results.items():
    print(f"חודש {month}: טמפ' ממוצעת = {stats['Avg_Temperature']:.2f}, לחות ממוצעת = {stats['Avg_Humidity']:.2f}, ממוצע גשם = {stats['Avg_Rain']:.2f}")