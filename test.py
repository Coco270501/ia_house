import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_california_housing

print('Colonnes', fetch_california_housing().feature_names)

df = fetch_california_housing(as_frame=True)['frame']
df.head()

X = df.drop(['MedHouseVal'], axis=1)
y = df['MedHouseVal']
X.shape, y.shape

# Diviser le jeu de données dans un rapport 80/20

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

from sklearn import preprocessing

std_scaler = preprocessing.StandardScaler()
X_train = std_scaler.fit_transform(X_train.astype(float))
X_test = std_scaler.transform(X_test.astype(float))

model = Sequential([
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])

model.fit(x=X_train, y=y_train, batch_size=128, epochs=5)
model.summary()

losses = pd.DataFrame(model.history.history)
losses.plot(figsize=(12, 8))

model.evaluate(X_test, y_test)

y_pred = model.predict(X_test)
y_pred

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))
print('MSE:', metrics.mean_squared_error(y_test, y_pred))
print('RSME:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('Varscore:', metrics.explained_variance_score(y_test, y_pred))

fig = plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred)

plt.plot(y_test, y_test, 'r')

predictions = model.predict(X_test[:1])
for i, prediction in enumerate(predictions):
    print(f'{i+1}: probability of price is: {prediction[0]} ')
    print(max(y))
