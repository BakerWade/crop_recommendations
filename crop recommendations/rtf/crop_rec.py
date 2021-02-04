import pandas as pd
import numpy as np

crop = pd.read_csv('Crop_recommendation.csv')

from sklearn.model_selection import train_test_split

X = crop.drop(['label'],axis=1)
y = crop['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=45)

from sklearn.tree import DecisionTreeClassifier

Dtree = DecisionTreeClassifier()
Dtree.fit(X_train,y_train)
pred = Dtree.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report

print(classification_report(y_test, pred))
print(confusion_matrix(y_test,pred))

from sklearn.ensemble import RandomForestClassifier

ran = RandomForestClassifier(n_estimators=200)

ran.fit(X_train, y_train)

prediction = ran.predict(X_test)

"""print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))"""