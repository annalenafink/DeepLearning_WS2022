#############################################
# AutoRegressive Integrated Moving Average. #
# Arima                                     #
#############################################
#Import the librarys
import math
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
#get Data from yahoo
data = yf.download('GOOGL', '2018-01-01', '2022-12-20')
df= data.iloc[:,[4]]
df1=np.log(df)

#Split the Dataframe into Train and Test data
len_df1= round(len(df1)*0.7)
train=df1.iloc[:len_df1]
test=df1.iloc[len_df1:]

#test if it is stationary
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
#calculate the value
p_value= adfuller(train)
print('p-value:', p_value[1])
##->proof that its not stationary


#Build the 1.difference to make it stationary
train_diff = train.diff().dropna()
#plot to compare
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
figure, axis = plt.subplots(2, 3)
#Regular
axis[0, 0].plot(df1,linewidth=1.0)
plot_acf(train, ax=axis[0, 1],linewidth=1.0)
plot_pacf(train,ax=axis[0, 2],linewidth=1.0)
#first difference
axis[1,0].plot(train_diff,linewidth=1.0)
plot_acf(train_diff,ax=axis[1,1],linewidth=1.0)
plot_pacf(train_diff,ax=axis[1,2],linewidth=1.0)
#calculate the value
p_value2= adfuller(train_diff)
print('p-value:', p_value2[1])

#Build the Arima Model
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

model = ARIMA(train, order=(1,1,1))#put the original model because this method is doing the difference for us (d=1)
model_fit = model.fit()
print(model_fit.summary())

#Make the prediction with the Arima model
forecast_test = model_fit.forecast(len(test))
df1['forecast'] = [None]*len(train) + list(forecast_test)
df1.plot()

#fit the ARIMA model automatically
import pmdarima as pm
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
auto_arima = pm.auto_arima(train, stepwise=False, seasonal=False)
auto_arima.summary()
#plot
auto_forecast = auto_arima.predict(n_periods=len(test))
df1['Auto Forecast'] = [None]*len(train) + list(auto_forecast)
df1.plot()
#Check the performance of the Model
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
#Mean Percentage Absolute Error
MAE=(metrics.mean_absolute_error(test, forecast_test)/test.mean())*100
print('Mean Absolute Error', MAE)

#Mean Squared Error
MSE=metrics.mean_squared_error(test, forecast_test)
print('Mean Squared Error', MSE)
#Root Mean Square Error
RMSE= np.sqrt(metrics.mean_squared_error(test, forecast_test))
print('Root Mean Square Error',RMSE)
#Regression Score Function
R2=metrics.r2_score(test, forecast_test)
print('Regression Score Function',R2)

#Predict which values i could take
import pmdarima as pm
from pmdarima.arima import auto_arima
import warnings
warnings.filterwarnings('ignore')
model_autoARIMA = auto_arima(y = train['Adj Close'], start_p=0,
                             start_q=0,
                             test='adf',
                             max_p=3,
                             max_q=3,
                             m=1,
                             d=None,
                             start_P=0,
                             D=0,
                             trace=True,
#                              error_action='ignore',
#                              suppress_warnings=True,
                             stepwise=True)
print(model_autoARIMA.summary())

#auto_forecast = model_autoARIMA.predict(n_periods=len(test))
#df1['Auto Forecast'] = [None]*len(train) + list(auto_forecast)
#df1.plot()
