import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    import matplotlib.pyplot as plt
    import numpy as np

    months = list(results.keys())
    avg_temps = [results[m]['Avg_Temperature'] for m in months]
    color_map = plt.cm.plasma  # או כל פלטת צבעים שתרצי

    colors = ['#A8DADC', '#F4A261', '#B7E4C7']  # תכלת, כתום-ורוד בהיר, ירוק עדין
    bar_colors = [colors[i % len(colors)] for i in range(len(months))]

    plt.figure(figsize=(12, 6))
    bars = plt.bar(months, avg_temps, color=color_map(np.linspace(0, 1, len(months))))

    plt.title('ממוצע טמפרטורות חודשי - גרף עמודות')
    plt.xlabel('חודש')
    plt.ylabel('טמפרטורה ממוצעת (°C)')
    plt.xticks(months)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # הוספת ערכים מעל העמודות
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval:.1f}', ha='center', va='bottom')

    plt.show()
