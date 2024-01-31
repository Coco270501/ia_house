import pandas as pd
from sklearn import metrics
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
# from sklearn.model_selection import train_test_split

from tensorflow import keras

data = pd.read_csv('housing.csv')

#Modifier les valeurs null
# data['Age'].fillna(data['Age'].median(), inplace=True)
# data['Embarked'].fillna('S', inplace=True)
# data['Cabin'].fillna('C00', inplace=True)

#Rajoutes des column selon les valeurs des colonnes pour les valeurs indiquées
data = pd.get_dummies(data, columns=['ocean_proximity'],dtype=np.float32)
print(data)

def min_max_scaling(series):
    return (series - series.min()) / (series.max() - series.min())

for col in data.columns:
    data[col] = min_max_scaling(data[col])

print(data)
features = data[['longitude','latitude','housing_median_age', 'total_rooms','total_bedrooms','population','households','median_income','ocean_proximity_<1H OCEAN', 'ocean_proximity_INLAND','ocean_proximity_ISLAND','ocean_proximity_NEAR BAY','ocean_proximity_NEAR OCEAN']]
labels = data['median_house_value']
train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size=0.2)
print(labels)
# print(train_features)
test_labels = train_test_split(features, labels, test_size=0.2)
# print(np.asarray(train_features).astype('float32'))
# #Construction
# #Dense toutes les couches sont liés entre elles
# #relu -> somme des poids des noeuds
# #train features 80% des donnees 1 ligne features (1 row et shape -1 dimension du vecter nb de colonnes) 
# #entree nb de colonne des featurs -> sortie nb de colonne des label
model = keras.Sequential(
    [
        keras.layers.Dense(15, activation='relu', input_shape=(train_features.shape[-1],)),
        #Ajout d'une couche avec 10 noeuronnes et connexion dense entre la couche 2 et la nouvelle couche
        keras.layers.Dense(10, activation='relu'),
        #Un seul neuronne -> la proba de savoir si survécu -> 99% -> 99% de chance de survit
        keras.layers.Dense(1)
    ]
)
#adam exploration de données et modification des poids
#accuracy 0.99 - 1 = 0.01 proche de 0 donc bonne accuracy pas de modification de poids avec l'optimizer adam
model.compile(optimizer='Adam', loss='mse', metrics=['mse', 'mae'])

# # #epochs durée de l'entrainement -> plus poches long plus entrainé mais peut perdre en précision si trop long
import numpy as np
history = model.fit(
    x=train_features,
    y=train_labels,
    epochs=20,
    batch_size = 400)
model.summary()

# test_loss, test_accuracy = model.evaluate(train_features,train_labels)
# print(f'Test accuracy {test_accuracy}')

predictions = model.predict(test_features[:5])
print("prediction on the first five test sample :")
for i, prediction in enumerate(predictions):
    print(f'{i+1}: probability of survival: {prediction[0]} ')
    # Actual: {test_labels}
