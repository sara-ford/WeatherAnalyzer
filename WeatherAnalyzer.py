import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# טוענים את הנתונים
df = pd.read_csv('weather_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# פונקציה להוספת עונה לפי חודש
def month_to_season(month):
    if month in [12, 1, 2]:
        return 'חורף'
    elif month in [3, 4, 5]:
        return 'אביב'
    elif month in [6, 7, 8]:
        return 'קיץ'
    else:
        return 'סתיו'

df['Season'] = df['Month'].apply(month_to_season)

# חישוב ממוצעים חודשיים
def monthly_averages(df):
    df['month'] = df['Date'].dt.month
    monthly_avg = {}
    for month in range(1, 13):
        month_data = df[df['month'] == month]

        avg_temp = np.mean(month_data['Temperature'])
        avg_humidity = np.mean(month_data['Humidity'])
        avg_rain = np.mean(month_data['Rain'])

        monthly_avg[month] = {
            'Avg_Temperature': avg_temp,
            'Avg_Humidity': avg_humidity,
            'Avg_Rain': avg_rain
        }
    return monthly_avg

results = monthly_averages(df)

# הדפסת התוצאות
for month, stats in results.items():
    print(f"חודש {month}: טמפ' ממוצעת = {stats['Avg_Temperature']:.2f}, לחות ממוצעת = {stats['Avg_Humidity']:.2f}, ממוצע גשם = {stats['Avg_Rain']:.2f}")

# גרף 1 - טמפרטורה ממוצעת חודשי (Matplotlib)
months = list(results.keys())
avg_temps = [results[m]['Avg_Temperature'] for m in months]
color_map = plt.cm.plasma

plt.figure(figsize=(12, 6))
bars = plt.bar(months, avg_temps, color=color_map(np.linspace(0, 1, len(months))))

plt.title('Monthly Average Temperatures')
plt.xlabel('Month')
plt.ylabel('Average Temperature (°C)')
plt.xticks(months)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.3, f'{yval:.1f}', ha='center', va='bottom')

plt.show()

# גרף 2 - boxplot עם נקודות (Seaborn)
plt.figure(figsize=(8,6))

sns.boxplot(x='Season', y='Temperature', data=df, order=['חורף', 'אביב', 'קיץ', 'סתיו'], palette='coolwarm')
sns.stripplot(x='Season', y='Temperature', data=df, order=['חורף', 'אביב', 'קיץ', 'סתיו'],
              color='black', size=3, jitter=True, alpha=0.5)

plt.title('השוואת טמפרטורות לפי עונות השנה עם נקודות')
plt.xlabel('עונה')
plt.ylabel('טמפרטורה (°C)')
plt.show()

# ניבוי עונה לפי טמפרטורה ולחות בעזרת Logistic Regression

# בחירת תכונות ומטרה
X = df[['Temperature', 'Humidity']].values
y = df['Season'].values

# המרת עונות למספרים
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# חלוקה לנתוני אימון ובדיקה
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# יצירת מודל ואימון
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ניבוי על נתוני הבדיקה
y_pred = model.predict(X_test)

# הצגת תוצאות
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# דוגמת ניבוי חדשה
new_sample = np.array([[25.0, 60.0]])  # טמפ' 25°, לחות 60%
predicted_season = le.inverse_transform(model.predict(new_sample))
print("ניבוי עונה לדוגמה:", predicted_season[0])
