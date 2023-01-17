########################################################################
#   Seminar - Deep Learning
#   Project: Visual Time Series Forecasting
#
#   Zeinab Saad
########################################################################
# Part1
import math
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
#Extract the Data from yahoo finance
data = yf.download('GOOGL', '2018-01-01', '2022-12-20')

#Display the Close Price History
plt.figure(1,figsize=(16,8))
plt.title('Adjusted Close Price History')
plt.plot(data['Adj Close'], linewidth=1.0)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adjusted Close Price', fontsize=18)
plt.show()

#Split the Dataframe
X = data.iloc[:,[0,1,2,3,5]]
y = data.iloc[:,[4]]
#Split the dataset into Training and Test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0,shuffle=False)

#Scaling data for better performance and the algorithm will understand our data better
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
XS_train = scaler.fit_transform(X_train)
XS_test = scaler.fit_transform(X_test)
ys_train = scaler.fit_transform(y_train)
ys_test = scaler.fit_transform(y_test)

#Reshape the Data for LSTM. The Input should be a 3D-Vector
XS_train=XS_train.reshape(XS_train.shape[0],XS_train.shape[1],1)

#Build the LSTM model
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

LSTM_model = Sequential()
LSTM_model.add(LSTM(125, return_sequences=True,recurrent_activation='relu', input_shape=(XS_train.shape[1], 1)))
LSTM_model.add(Dropout(0.3)) #to prevent overfitting
LSTM_model.add(LSTM(125, return_sequences=True,recurrent_activation='relu'))
LSTM_model.add(Dropout(0.3)) #to prevent overfitting
LSTM_model.add(LSTM(125, return_sequences=False))#false because we will not add more lstm layers
LSTM_model.add(Dense(1,activation='relu'))


###################################################################################
# compile the model                                                               #
# Adam optimization is a stochastic gradient descent method                       #
# that is based on adaptive estimation of first-order and second-order moments    #
# Loss function is used to find error or deviation in the learning process.       #
# Keras requires loss function during model compilation process.                  #
###################################################################################
LSTM_model.compile(optimizer='adam', loss='mean_squared_error')
LSTM_model.summary()

#Train the LSTM-Model
train = LSTM_model.fit(XS_train,ys_train,validation_data = (XS_test, XS_test),epochs=300,batch_size=64,shuffle=True)#64

#to evaluate how well our modell performs on unseen data
LSTM_model.evaluate(XS_test, ys_test)
#Evaluate the loss/performance
plt.figure(2,figsize=(16,8))
axes=plt.axes()
axes.plot(pd.DataFrame(train.history)['loss'], label='Loss')
axes.plot(pd.DataFrame(train.history)['val_loss'], label='Validation Loss')
axes.legend(loc=0)
axes.set_title('Model fitting performance')
plt.show()

#Predict Data/output value
XS_test = XS_test.reshape((XS_test.shape[0], XS_test.shape[1], 1))
predicted_values = LSTM_model.predict(XS_test)
predicted_values = scaler.inverse_transform(predicted_values)
predicted_values.shape
#Check the performance of the Model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
NX_train=scaler.inverse_transform(LSTM_model.predict(XS_train))
Ny_train=scaler.inverse_transform(ys_train.reshape(ys_train.shape[0],1))
Ny_test=scaler.inverse_transform(ys_test)
###########################
# Prediction performance  #
###########################
#Mean Percentage Absolute Error
MAE=(metrics.mean_absolute_error(Ny_test, predicted_values)/Ny_test.mean())*100
print('Mean Absolute Error', MAE)

#Mean Squared Error
MSE=metrics.mean_squared_error(Ny_test, predicted_values)
print('Mean Squared Error', MSE)
#Root Mean Square Error
RMSE= np.sqrt(metrics.mean_squared_error(Ny_test, predicted_values))
print('Root Mean Square Error',RMSE)
#Regression Score Function
R2=metrics.r2_score(Ny_test, predicted_values)
print('Regression Score Function',R2)

#Comparing the labeled data
plt.figure(3,figsize=(16,8))
gr=plt.axes()
plt.plot(Ny_test, linewidth=1.0,label='True label')
plt.plot(predicted_values,marker = '.', linewidth= 1.0,label='Predicted label')
gr.legend(loc=0)
plt.show()

##Prepare the Dataframe for the plot
y_len = round(len(y)*0.7)-1 #length of training data y
true_data= data.iloc[:y_len]#true
trained_data= data.iloc[y_len:] #trained
trained_data['Predictions'] = predicted_values
compare= trained_data[['Adj Close','Predictions']]

#Plot of the Visual Time Series Forecasting
plt.figure(4,figsize=(15,8))
plt.title('Visual Time Series Forecasting')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Adjusted Close Price', fontsize=18)
plt.plot(true_data['Adj Close'],linewidth=3)
plt.plot(trained_data[['Adj Close']],linewidth=3,color='g')
plt.plot(trained_data[['Predictions']],linewidth=3,color='y')
plt.legend(['True', 'Valid', 'Predictions'], loc='lower right')
plt.show()