#generate deep learning model fo gait analysis with numerical data

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score

#load data
data = pd.read_csv('synthetic_dataset.csv')
data.head()
x = data[["R Stance","L Stance","R Swing","L Swing","R Double Supp.","L Double Supp.","R Step Length","L Step Length","R Speed","L Speed","R Hip Rom","L Hip Rom","R Knee Rom","L Knee Rom","R Ankle Rom","L Ankle Rom","Cadence","Step Width"]]
y = data['Target']
print(x.shape)
print(y.shape)

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.179, random_state=42)
#train_features.describe().transpose()[['mean', 'std']]
#test_features.describe().transpose()[['mean', 'std']]
#sns.pairplot(train_features[['x1', 'x2', 'x3', 'x4', 'x5']], diag_kind='kde')
#plt.show()

#normalization
normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(x_train))
#print(normalizer.mean.numpy())
#first = np.array(train_features[:1])
#with np.printoptions(precision=2, suppress=True):


#build model
model = keras.Sequential([
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
#print(model.summary())

#compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
#train model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    verbose=0, epochs=7)
#print(history.history.keys())
#plot loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')

#plot accuracy
#plt.plot(history.history['accuracy'])
#plt.plot(history.history['val_accuracy'])
#plt.title('model accuracy')
#plt.ylabel('accuracy')

#evaluate model
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=2)
print('\nTrain accuracy:', train_acc)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)
#make prediction
predictions = model.predict([[66.6,61.6,33.4,38.4,13.9,12.8,0.35,0.4,0.6,0.62,37.3,36.73,50.18,44.48,14.33,12.65,89.85,0.17]])
print(int(predictions))

#classification report
