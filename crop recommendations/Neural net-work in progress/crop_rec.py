import numpy as np
from numpy.lib.shape_base import apply_over_axes
import pandas as pd
from pandas.core.indexing import convert_to_index_sliceable
import seaborn as sns 
import matplotlib.pyplot as plt
from seaborn.relational import lineplot
from tensorflow.keras import callbacks

crop = pd.read_csv('Crop_recommendation.csv')
#print(crop['label'].unique())

#For all work done with visual representation, the average of each column per unique label was used

#created a number of series depicting the mean of each column in "crop" per label

"""rice = crop[crop['label']=='rice'].mean()
maize = crop[crop['label']=='maize'].mean()
chickpea = crop[crop['label']=='chickpea'].mean()
kidneybeans = crop[crop['label']=='kidneybeans'].mean()
pigeonpeas = crop[crop['label']=='pigeonpeas'].mean()
mothbeans = crop[crop['label']=='mothbeans'].mean()
mungbean = crop[crop['label']=='mungbean'].mean()
blackgram = crop[crop['label']=='blackgram'].mean()
lentil = crop[crop['label']=='lentil'].mean()
pomegranate = crop[crop['label']=='pomegranate'].mean()
banana = crop[crop['label']=='banana'].mean()
mango = crop[crop['label']=='mango'].mean()
grapes = crop[crop['label']=='grapes'].mean()
watermelon = crop[crop['label']=='watermelon'].mean()
muskmelon = crop[crop['label']=='muskmelon'].mean()
apple = crop[crop['label']=='apple'].mean()
orange = crop[crop['label']=='orange'].mean()
papaya = crop[crop['label']=='papaya'].mean()
coconut = crop[crop['label']=='coconut'].mean()
cotton = crop[crop['label']=='cotton'].mean()
jute = crop[crop['label']=='jute'].mean()
coffee = crop[crop['label']=='coffee'].mean()

#DataFrame containing all the series in it

crop2 = crop.drop(['label'], axis=1).columns
crop3 = crop['label'].unique()
avg_label = pd.DataFrame([rice, maize, chickpea, kidneybeans,pigeonpeas,mothbeans, mungbean,blackgram,lentil,pomegranate, banana,
                        mango,grapes,watermelon,muskmelon,apple,orange,papaya,coconut,cotton,jute,coffee],index= crop3, columns= crop2)

avg_label.index.names =['Crop']
avg_label.columns.names= ['factors']"""

"""avg_label.plot.bar(stacked=True)
plt.xlabel('Crop')
plt.title('Mean factors for crops')


#pairplot, depicting the columns of avg_labels VS the different crop types
label = crop3
n = avg_label['N']
p = avg_label['P']
k = avg_label['K']
temp = avg_label['temperature']
humid = avg_label['humidity']
ph = avg_label['ph']
rain = avg_label['rainfall']

fig, axs = plt.subplots(4,2, figsize=(12,9), sharex=True)

axs[0,0].plot(label,n, color='green')
axs[0,0].set_ylabel('Nitrogen/N')
axs[0,0].set_facecolor('black')

axs[0,1].plot(label,p, color='white')
axs[0,1].set_ylabel('Phosphorus/P')
axs[0,1].set_facecolor('black')

axs[1,0].plot(label,k, color='grey')
axs[1,0].set_ylabel('Potassium/K')
axs[1,0].set_facecolor('black')

axs[1,1].plot(label,temp, color='red')
axs[1,1].set_ylabel('Temp')
axs[1,1].set_facecolor('black')

axs[2,0].plot(label,humid, color='orange')
axs[2,0].set_ylabel('Humidity')
axs[2,0].set_facecolor('black')

axs[2,1].plot(label,ph, color='tab:pink')
axs[2,1].set_ylabel('ph level')
axs[2,1].set_facecolor('black')

axs[3,0].plot(label,rain, color='blue', alpha=0.4)
axs[3,0].tick_params(axis='x', rotation=90)
axs[3,0].set_ylabel('Rain')
axs[3,0].set_facecolor('black')

axs[3,1].set_title('Combined graph')
axs[3,1].plot(label,n, color='green')
axs[3,1].plot(label,p, color='white')
axs[3,1].plot(label,k, color='grey')
axs[3,1].plot(label,temp, color='red')
axs[3,1].plot(label,humid, color='orange')
axs[3,1].plot(label,ph, color='tab:pink')
axs[3,1].plot(label,rain, color='blue', alpha=0.4)
axs[3,1].tick_params(axis='x', rotation=90)
axs[3,1].set_facecolor('grey')

g = sns.PairGrid(avg_label, palette='tab10')
g.map_diag(plt.hist)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)
plt.show()"""

dummies = pd.get_dummies(crop['label'])
crop = pd.concat([crop.drop('label',axis=1),dummies],axis=1)

#print(crop[crop['apple']==1])


from sklearn.model_selection import train_test_split

X = crop.drop(['apple','banana',
       'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes',
       'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
       'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
       'pomegranate', 'rice', 'watermelon'],axis=1).values
y = crop[['apple','banana',
       'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'grapes',
       'jute', 'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans',
       'mungbean', 'muskmelon', 'orange', 'papaya', 'pigeonpeas',
       'pomegranate', 'rice', 'watermelon']].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

model = Sequential()
model_load = load_model('crop_prediction_model.h5')
"""model.add(Dense(7,activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(21,activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(42,activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(42,activation='relu'))
#model.add(Dropout(0.5))

model.add(Dense(22,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)

model.fit(x=X_train,y=y_train, epochs=500, validation_data=(X_test,y_test), callbacks=[early_stop])"""

#model.save('crop_prediction_model.h5')

"""loss = pd.DataFrame(model.history.history)
loss.plot()
plt.show()"""



pred = model_load.predict(X_test)
pred = np.argmax(pred,axis=1)


from sklearn import metrics

score = metrics.accuracy_score(y_test,pred)
print('Accuracy score:'.format(score))




