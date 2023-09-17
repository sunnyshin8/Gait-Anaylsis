from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
df = pd.read_csv('synthetic_dataset1.csv')
print(df.head())
print(df.shape)
print(df.columns)  
X = df[['R Stance', 'L Stance', 'R Swing', 'L Swing',
       'R Double Supp.', 'L Double Supp.', 'R Step Length', 'L Step Length',
       'R Speed', 'L Speed', 'R Hip Rom', 'L Hip Rom', 'R Knee Rom',
       'L Knee Rom', 'R Ankle Rom', 'L Ankle Rom', 'Cadence', 'Step Width']]
X = X.values
Y = df['Target']
X_train,X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, Y_train)
acc0 = accuracy_score(Y_train, model.predict(X_train)) #training accuracy
acc1 = accuracy_score(Y_test, model.predict(X_test)) #testing accuracy
print(acc0, "\n",acc1)
print(model.predict([[63.831402034971184,64.26559477175036,36.198504253627725,35.691244312797735,14.558195656910348,13.579523645085745,0.39481112366286086,0.472684620404192,-0.14350083741156316,1.0927830585414047,41.85483832042118,49.79681810267824,36.7691551880132,54.83448148225083,31.151018426963688,29.463418114153782,69.67833554763052,0.4072933992303819]]))

