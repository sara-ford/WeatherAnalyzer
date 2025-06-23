import pandas as pd
import numpy as np
import datetime

# יצירת תאריכים לכל יום בשנת 2024
dates = pd.date_range(start='2024-01-01', end='2024-12-31')

# פונקציות ליצירת נתונים מזויפים בהתאם לחודש
def generate_temperature(date):
    month = date.month
    if month in [12, 1, 2]:  # חורף
        return np.random.randint(5, 15)
    elif month in [3, 4, 5]:  # אביב
        return np.random.randint(15, 25)
    elif month in [6, 7, 8]:  # קיץ
        return np.random.randint(25, 35)
    else:  # סתיו
        return np.random.randint(15, 25)

def generate_humidity(date):
    return np.random.randint(40, 90)

def generate_rain(date):
    month = date.month
    if month in [12, 1, 2]:  # חורף - יותר סיכוי לגשם
        return np.random.choice([0, 1], p=[0.6, 0.4])
    elif month in [3, 4, 5, 9, 10, 11]:
        return np.random.choice([0, 1], p=[0.8, 0.2])
    else:  # קיץ
        return 0

# יצירת הדאטה
data = {
    'Date': dates,
    'Temperature': [generate_temperature(d) for d in dates],
    'Humidity': [generate_humidity(d) for d in dates],
    'Rain': [generate_rain(d) for d in dates]
}

# הפיכת המילון לדאטהפריים
df = pd.DataFrame(data)

# שמירה לקובץ CSV
df.to_csv('weather_data.csv', index=False, encoding='utf-8')

print("✅ קובץ weather_data.csv נוצר בהצלחה!")
