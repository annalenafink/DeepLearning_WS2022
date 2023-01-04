########################################################################
#   Seminar - Deep Learning
#   Project: Visual Time Series Forecasting
#
#   Zeinab Saad
########################################################################
import math
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get the data from yahoo finance
tstart='2018-01-01'
tend='2022-12-20'
data = yf.download('GOOGL', tstart, tend)#period='5y', interval='1d')
#To avoid the following error message- AttributeError: 'DataFrame' object has no attribute 'Date'
data.reset_index(inplace=True)
#convert datetime to object
data['Date'] = data['Date'].dt.strftime("%Y-%m-%d")
#check info of dataframe
data.shape
data.info()
#Convert argument to datetime
data["Date"]=pd.to_datetime(data.Date,format="%Y-%m-%d")
data.index=data.pop('Date') #damit die x-achse die Jahreszahl anzeigt -> habe nun zwei spalten mit datum
#Plot the graph
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(data['Close'], linewidth=1.0)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Close Price', fontsize=18)
plt.show()
# Print the last 5 rows
print(data.tail())
#convert into numpy
data.values


###split the data
# get the locations
X = data.iloc[:, :-1] #the first five columns
y = data.iloc[:, -1] #the last column
print(X)
print(y)

# split the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train))#70% of the dataset
print(len(X_test))#30% of the dataset
print(len(y_test))#30%
print(len(y_train))#70%
#normalize the data, to improve the performance
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(np.array(X_train))
print(len(X_train))
X_test = scaler.fit_transform(np.array(X_test))
print(len(X_test))
y_train = scaler.fit_transform(np.array(y_train).reshape(-1,1))
print(len(y_train))
y_test = scaler.fit_transform(np.array(y_test).reshape(-1,1))
print(len(y_test))
#because of an error i had to reshape my dataset here
#error was: ValueError: Expected 2D array, got 1D array instead: ....
#reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.

type(X_test)

#################################################################
# reshape because fitting data into the model                   #
# made with the Keras framework needs a suitable                #
# dimensionality (input shapes acceptable by LSTM layer is 3D)  #
#################################################################
#1 := features
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)
y_test=y_test.reshape(y_test.shape[0],1)
y_train=y_train.reshape(y_train.shape[0],1)
type(X_train)

X_train=tensorflow.convert_to_tensor(X_train)
type(X_train)
X_test=tensorflow.convert_to_tensor(X_test)
y_train=tensorflow.convert_to_tensor(y_train)
y_test=tensorflow.convert_to_tensor(y_test)

X_train.shape
X_test.shape
y_train.shape
y_test.shape





#Build the LSTM-Model to do the prediction
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense, LSTM
from tensorflow.keras import Sequential

LSTM_model = tensorflow.keras.Sequential()
LSTM_model.add(LSTM(125,return_sequences=True,recurrent_activation='relu',input_shape=(X_train.shape[1],1)))
#LSTM_model.add(LSTM(50))
LSTM_model.add(Dense(1,activation='relu'))




###################################################################################
# compile the model                                                               #
# Adam optimization is a stochastic gradient descent method                       #
# that is based on adaptive estimation of first-order and second-order moments    #
# Loss function is used to find error or deviation in the learning process.       #
# Keras requires loss function during model compilation process.                  #
###################################################################################
from keras import losses
LSTM_model.compile(loss='mean_squared_error',optimizer='adam')
LSTM_model.summary()

#train the model with model.fit()
LSTM_model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64)#64

#prediction
train_predict=LSTM_model.predict(X_train)
test_predict=LSTM_model.predict(X_test)

#evaluate
LSTM_model.evaluate(X_test, y_test)

#convert back
NX_train = scaler.inverse_transform(np.array(X_train).reshape(X_train.shape[0],X_train.shape[1]))
print(NX_train)
NX_test = scaler.inverse_transform(np.array(X_test).reshape(X_test.shape[0],X_test.shape[1]))
print(NX_test)
Ny_test = scaler.inverse_transform(np.array(y_test))
print(Ny_test)
Ny_train = scaler.inverse_transform(np.array(y_train))
print(Ny_train)
#oder kürzer geht auch scaler.inverse_transform(np.array(X_train).reshape(-1,1))


#plot new graph
#Part1
plt.figure(figsize=(16,8))
plt.plot(Ny_test, linewidth=1.0,color='b')
plt.plot(Ny_train, linewidth=1.0,color='r')
plt.plot(NX_train, linewidth=1.0,color='c')
plt.plot(NX_test, linewidth=1.0,color='g')
#data.index

trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
Note: Your results may vary given the stoc