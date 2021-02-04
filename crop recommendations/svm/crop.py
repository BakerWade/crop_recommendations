import pandas as pd
import numpy as np

crop = pd.read_csv('Crop_recommendation.csv')

from sklearn.model_selection import train_test_split

X= crop.drop(['label'], axis=1)
y = crop['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix

"""print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))"""

#lets see if doing a gridsearch will affect this data

from sklearn.model_selection import GridSearchCV
param = {'C':[0.001,0.1,1,10,100], 'gamma':[10,1,0.1,0.001,0.0001,0.00001]}
grid= GridSearchCV(SVC(), param, refit=True, verbose=3)
grid.fit(X_train, y_train)

prediction = grid.predict(X_test)

print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))