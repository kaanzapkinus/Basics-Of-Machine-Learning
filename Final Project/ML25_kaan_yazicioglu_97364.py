import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

#1 DATASET & CONTROLS

#import
file_path = 'antalyarentals.csv'
df = pd.read_csv(file_path)

#basic infos
print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset shape:", df.shape)

#checking for missing values. (no missing values)
print("\nMissing values in each column:")
print(df.isnull().sum())

#2 DATA PREPROCESING

#i am removing flats over 50000 tl from dataset because they are so expensive and causes bad stats for models
df = df[df['rent_price'] <= 50000]

#converting 'rooms' column to integer (example "3+1" = 4)
def parse_rooms(value):
    try:
        return sum(int(x) for x in value.split('+'))
    except:
        return np.nan

#converting 'building_age' column to approximate number (example "between 21-25 " = 23) 
def parse_age(value):
    text = str(value).lower()
    if text == '0':
        return 0
    elif '2-4' in text:
        return 3
    elif '5-10' in text:
        return 7.5
    elif '11-15' in text:
        return 13
    elif '16-20' in text:
        return 18
    elif '21-25' in text:
        return 23
    elif '26-30' in text:
        return 28
    elif '31' in text:
        return 35
    else:
        return np.nan

#seperate apartments as new and old by year.
df['building_age'] = pd.to_numeric(df['building_age'], errors='coerce')
df['is_new_building'] = df['building_age'].apply(lambda x: 1 if x < 15 else 0)

#applying the numbers to dataset
df['rooms'] = df['rooms'].apply(parse_rooms)
df['building_age'] = df['building_age'].apply(parse_age)

#3 MODEL IMPLEMENTATION & TRAINING

# Selected input features (X):
# - rooms: total number of rooms (example 3+1 becomes 4)
# - is_new_building: binary label for building condition (1 = newer than 15 years, 0 = 15 years or older)
# - net_m2: usable/net area in square meters
# - elevator: whether the apartment has an elevator (0 or 1)
# - compound: whether the apartment is in a residential compound (0 or 1)
# - fee: monthly maintenance fee
# - furnished: whether the apartment is furnished (0 or 1)
 
print("\nMissing values in selected features:")
print(df[['rooms', 'is_new_building', 'net_m2', 'elevator', 'compound', 'fee', 'furnished']].isnull().sum())

#dropping rows with missing 'rooms' or 'is_new_building' values
df = df.dropna(subset=['rooms', 'is_new_building'])

#Linear Regression

features = ['rooms', 'is_new_building', 'net_m2', 'elevator', 'compound']
X = df[features]
y = df['rent_price']

#split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=35
)

#train the model
model = LinearRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\nModel performance:")
print(f"Train R² score: {train_score:.3f}")
print(f"Test R² score: {test_score:.3f}")

#Random Forest Regression

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

features = ['rooms', 'is_new_building', 'net_m2', 'elevator', 'compound', 'fee', 'furnished']
X = df[features]
y = df['rent_price']

#dropping missing values in selected features
df = df.dropna(subset=features)

#split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

rf_model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=35)
rf_model.fit(X_train, y_train)

#scores
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

train_score = r2_score(y_train, y_train_pred)
test_score = r2_score(y_test, y_test_pred)

print("\nRandom Forest Model performance:")
print(f"Train R² score: {train_score:.3f}")
print(f"Test R² score: {test_score:.3f}")

import matplotlib.pyplot as plt

#feature importance for random forest model
importances = rf_model.feature_importances_
plt.barh(features, importances)
plt.xlabel("Importance")
plt.title("Feature Importance (Random Forest)")
plt.show()

#4 ANALYSIS AND MODEL EVALUATION

# Tahmin yap
y_pred = rf_model.predict(X_test)

# Hataları hesapla
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nError Analysis:")
print(f"Mean Absolute Error (MAE): {mae:.2f} TL")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f} TL")

# Hata dağılımı görselleştirme
errors = y_test - y_pred

plt.figure(figsize=(10, 5))
plt.hist(errors, bins=30, color='salmon', edgecolor='black')
plt.title('Prediction Error Distribution')
plt.xlabel('Prediction Error (TL)')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.show()

plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Rent Price')
plt.ylabel('Predicted Rent Price')
plt.title('Actual vs Predicted Rent Prices')
plt.show()