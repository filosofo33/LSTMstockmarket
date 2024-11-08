
# coding: utf-8


# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


# In[2]:

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    data = sci_minmax(data)
    
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    
    result = np.array(result)#results solo cada secuencia
    print(result.shape)
    row = round(0.9 * result.shape[0])#90% train
    train = result[:int(row), :]
    x_train = train[:, :-1]#tos los casos pero el 20avo no
    #print(train[:, :-1])
    #print(train[:, :-1].shape)
    #print("xxxxxxx")
    #print(train[:, -1])
    #print(train[:, -1].shape)
    #print("yyyyyyyyyyyyyyyy")
    #print(train[:, -1][:,1])
    #print(train[:, -1][:,1].shape)
    y_train = train[:, -1][:,1]#el 20avo es la respuesta,y el valor dl stock el 1
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]


# In[3]:

#normalization
#min max scaler normalization
def numpy_minmax(X):#customizada el maximo
    xmin =  X.min(axis=0)
    resultnorm=(X - xmin) / (X.max(axis=0) - xmin)
    return resultnorm
    #return (X - xmin) / (X.max(axis=0) - xmin)
def normalizeinput(Xmin,Xmax,inptest):#customizada el maximo
    #inptest[inptest>Xmax]=1#esta bien¿?
    #inptest[inptest<Xmin]=0#esta bien¿?
    resultnorm=(inptest - Xmin) / (Xmax - Xmin)
    return resultnorm
    #return (X - xmin) / (X.max(axis=0) - xmin)
def sci_minmax(X):#separa por columnas la normalizacion
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1), copy=True)
    return minmax_scale.fit_transform(X)
def normaliz_labels(X):
    tamano=X.shape[0]
    for e in xrange(0,tamano):
        if(X[e]<minResultLabel):
            X[e]=minResultLabel
        if(X[e]>maxResultLabel):
            X[e]=maxResultLabel
    return (X/maxResultLabel)

def classific_normaliz(X):
    tamano=X.shape[0]
    for e in xrange(0,tamano):
        if(X[e]>=2.84):#2.84,3.69
            X[e]=1
        elif(X[e]<=-1.18):
            X[e]=-1
        #elif(X[e]>=0.12):
        #    X[e]=2
        else:
            X[e]=0
    return (X)

def labeltodimensions(labeling):
    sizel=labeling.shape[0]
    resultl=np.zeros([sizel,sizelabels])
    g1=0
    g2=0
    g3=0
    g4=0
    for i in range (0,sizel):
        if(labeling[i]==1):
            g1=g1+1
            resultl[i,0]=1
        elif(labeling[i]==-1):
            g2=g2+1
            resultl[i,2]=1
        #elif(labeling[i]==2):
        #    g3=g3+1
        #    resultl[i,0]=0.5
        else:#3clases
            g4=g4+1
            resultl[i,1]=1
    print("grupos %f %f %f %f" % (g1,g2,g3,g4))
    return resultl


# In[4]:

df = pd.read_csv('7m5.csv')#, names=['fecha','idtienda','idproducto','ventas']
print(df.shape)
print(df.columns[1])
#df[['Fecha','Ventas']].set_index('Fecha').plot()
#df.ix[1111:1112,0:76]

#Remove inalterable columns

cgot=[]

print(df["symbol"].unique())
for i in range (0,df.shape[1]):
    if(df[str(df.columns[i])].unique().size!=1):
        cgot.append(str(df.columns[i]))

date_splita = df['datetime'].str.split(' ').str[0]
df['datetime']=date_splita
date_split = df['datetime'].str.split('-').str
df['Year'], df['Month'], df['Day'] = date_split
resultdays = [int(i) for i in df['Day']]
df['datetime']=resultdays
xinputs=df.ix[:,cgot]

xinputsm=xinputs.as_matrix()
xminim= xinputsm.min(axis=0)
xmaxim= xinputsm.max(axis=0)

#Initializing parameters
window = 26
sizefeatures = len(xinputs.columns)

#print(xinputs[1:5])



X_train, y_train, X_test, y_test = load_data(xinputs, window)
print("X_train", X_train.shape)
print("y_train", y_train.shape)
print("X_test", X_test.shape)
print("y_test", y_test.shape)
print(X_train[0])
print(y_train[0])

#window will be 26, 3 parts


# In[5]:

#Building models

def build_model(layers):
    #0.0017val test 
    d = 0.2
    model = Sequential()
    model.add(LSTM(228, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(164, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dense(16,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model

def build_model2(layers):
        #0.0015VAL #0.0077
        d = 0.2
        model = Sequential()
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model


def build_model4(layers):
        #0.0015VAL
        d = 0.2
        model = Sequential()
        model.add(LSTM(603, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(300, input_shape=(layers[1], layers[0]), return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model

def build_model3(layers):
    #0.0011VAL test 0.0038
    d = 0.2
    model = Sequential()
    model.add(LSTM(228, input_shape=(layers[1], layers[0]), return_sequences=True))
    model.add(Dropout(d))
    model.add(LSTM(164, input_shape=(layers[1], layers[0]), return_sequences=False))
    model.add(Dropout(d))
    model.add(Dense(64,init='uniform',activation='relu'))        
    model.add(Dense(25,init='uniform',activation='relu'))        
    model.add(Dense(1,init='uniform',activation='linear'))
    model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    return model


# In[ ]:

#model = build_model2([sizefeatures,window,1])
#model = build_model2([sizefeatures,window,1])
model = build_model3([sizefeatures,window,1])


# In[ ]:

model.fit(
    X_train,
    y_train,
    batch_size=1512,
    nb_epoch=400,
    validation_split=0.15,
    verbose=1)


# In[ ]:

#trainScore = model.evaluate(X_train, y_train, verbose=0)
#print('Train Score: %f MSE (%f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

testScore = model.evaluate(X_test, y_test, verbose=0)
print('Test Score: %f MSE (%f RMSE)' % (testScore[0], math.sqrt(testScore[0])))


# In[ ]:




# In[ ]:




# In[ ]:

# print(X_test[-1])
diff=[]
ratio=[]
p = model.predict(X_test)
for u in range(len(y_test)):
    pr = p[u][0]
    ratio.append((y_test[u]/pr)-1)
    diff.append(abs(y_test[u]- pr))
    #print(u, y_test[u], pr, (y_test[u]/pr)-1, abs(y_test[u]- pr))


# ## Predictions vs Real results

# In[ ]:

import matplotlib.pyplot as plt2

plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='y_test')
plt2.legend(loc='upper right')
plt2.show()


# In[ ]:




# In[ ]:

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
#json_file = open('model.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json)
# load weights into new model
#loaded_model.load_weights("model.h5")
#print("Loaded model from disk")
 
# evaluate loaded model on test data
#oaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
#print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[ ]:



