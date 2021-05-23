###
### PRAHITHA MOVVA
### S20180010108
###

import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels
import datetime
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.api import VAR
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from scipy import stats
from sklearn.metrics import mean_squared_error

class StationarityTests:
    def __init__(self, significance=0.05):
        self.SignificanceLevel = significance
        self.pValue = None
        self.isStationary = None

    # if p-value less than the significance level, reject the null hypothesis
    # which means the time series is stationary
    def ADF_Stationarity_Test(self, timeseries, printResults=True):
        # dickey-fuller test
        adfTest = adfuller(timeseries, autolag='AIC')
        self.pValue = adfTest[1]

        if(self.pValue < self.SignificanceLevel):
            self.isStationary = True
        else:
            self.isStationary = False

        if printResults:
            dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic', 'P-Value', '# Lags used', '# Observations Used'])
            # add critical values
            for key, value in adfTest[4].items():
                dfResults['Critical Value (%s)'%key] = value
            print('Augmented Dickey-Fuller Test Results:')
            print(dfResults)

# import the data
uv_data1 = pd.read_csv("./GNFUV USV Dataset/pi2/gnfuv-temp-exp1-55d487b85b-5g2xh_1.0.csv", header=None)
uv_data2 = pd.read_csv("./GNFUV USV Dataset/pi3/gnfuv-temp-exp1-55d487b85b-2bl8b_1.0.csv", header=None)
uv_data3 = pd.read_csv("./GNFUV USV Dataset/pi4/gnfuv-temp-exp1-55d487b85b-xcl97_1.0.csv", header=None)
uv_data4 = pd.read_csv("./GNFUV USV Dataset/pi5/gnfuv-temp-exp1-55d487b85b-5ztk8_1.0.csv", header=None)

# data cleaning
# manually changed the format to csv, rows were earlier in string format

# adding column names
uv_data1.columns = ['device', 'humidity', 'temperature', 'experiment', 'time']
uv_data2.columns = ['device', 'humidity', 'temperature', 'experiment', 'time']
uv_data3.columns = ['device', 'humidity', 'temperature', 'experiment', 'time']
uv_data4.columns = ['device', 'humidity', 'temperature', 'experiment', 'time']

warnings.filterwarnings("ignore")

###
### DEVICE 1
###
print("\n\nDevice 1\n\n")
print(uv_data1.shape)
print(uv_data1.info()) # check the number of rows, columns and type of each column
print(uv_data1.describe()) # this gives the count, unique, top, freq entries of the data

def conv_time(x):
    x = datetime.datetime.fromtimestamp(x)
    return x

uv_data1['time'] = uv_data1['time'].apply(conv_time)
data = uv_data1
uv_data1 = data.drop(['time', 'experiment'], axis=1)
uv_data1.index = data.time
uv_data1 = uv_data1.drop(['device'], axis=1)
# print(uv_data1)

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.suptitle('Multi dimensional time series (Device 1)')
for i, ax in enumerate(axes.flatten()):
    data = uv_data1[uv_data1.columns[i]]
    ax.plot(data)
    ax.set_title(uv_data1.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()

plt.title('Device 1')
main_model = uv_data1
sns.boxplot(x=main_model['temperature'], y=main_model['humidity'])
plt.show()

humidity = uv_data1['humidity']
temperature = uv_data1['temperature']
fig, axs = plt.subplots(2)
fig.suptitle('Histogram for humidity and temperature (Device 1)')
axs[0].hist(humidity, bins=15)
axs[0].set_title('Humidity')
axs[1].hist(temperature, bins=10)
axs[1].set_title('Temperature')
plt.show()

# humidity autocorrelation plot
autocorrelation_plot(uv_data1['humidity'])
plt.title('Humidity')
plt.show()

# temperature autocorrelation plot
autocorrelation_plot(uv_data1['temperature'])
plt.title('Temperature')
plt.show()

autocorrelation_plot(uv_data1[['humidity', 'temperature']])
plt.title('Humidity and Temperature')
pyplot.show()

# correlation
print('Correlation between humidity and temperature\n')
correlation = uv_data1[['humidity', 'temperature']].corr()
print(correlation)
# we see a negative correlation of -0.97, so we can say that they are inversely proportional

# to extract maximum information from the data
# it is important to have a normal or gaussian distribution of the data
# to check fot that, I have done a normality test based on null and
# alternate hypothesis solution
print('\nHumidity:')
stat, p = stats.normaltest(uv_data1.humidity)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print('\nTemperature:')
stat, p = stats.normaltest(uv_data1.temperature)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

# visualizing the skewness and kurtosis of the data
print('\nHumidity:\n')
sns.displot(uv_data1.humidity)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data1.humidity)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data1.humidity)))

# print('\nTemperature:\n')
sns.displot(uv_data1.temperature)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data1.temperature)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data1.temperature)))

# to check if each of the time-series in the system influences each other
# p-values less than significance level (0.05), implies that the hypothesis 
# X does not cause Y can be rejected
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

check_causation = grangers_causation_matrix(uv_data1, variables=uv_data1.columns)
print(check_causation)
# p value is less than significance level (0.05), so we can conclude that
# humidity and temperature are influencing each other

# to check whether or not they have unit roots by using the
# augmented dickey fuller (ADF) test
# ADF test is used to determine if the time series is stationary or not
# unit roots are a cause for the non-stationarity
sTest = StationarityTests()
print('\nHumidity:')
sTest.ADF_Stationarity_Test(uv_data1['humidity'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature:')
sTest.ADF_Stationarity_Test(uv_data1['temperature'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# from ADF Test, cannot reject null hypothesis that the series has a unit root
# so we try various methods like taking log, sqrt or difference
# to convert the series to stationary

# running on naive regression to observe what happens
# on the undifferenced time-series
uv_data1['const'] = 1
model1 = sm.OLS(endog=uv_data1['humidity'], exog=uv_data1[['temperature', 'const']])
result1 = model1.fit()
print('\n')
print(result1.summary())
plot_acf(result1.resid, alpha=0.05)
plt.show()


# transform the columns using natural log
uv_data1['humidity_log'] = np.log(uv_data1['humidity'])
uv_data1['temperature_log'] = np.log(uv_data1['temperature'])
sTest = StationarityTests()
print('\nHumidity Log:')
sTest.ADF_Stationarity_Test(uv_data1['humidity_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Log:')
sTest.ADF_Stationarity_Test(uv_data1['temperature_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# transform the columns using square root
uv_data1['humidity_sqrt'] = np.sqrt(uv_data1['humidity'])
uv_data1['temperature_sqrt'] = np.sqrt(uv_data1['temperature'])
sTest = StationarityTests()
print('\nHumidity Square Root:')
sTest.ADF_Stationarity_Test(uv_data1['humidity_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Square Root:')
sTest.ADF_Stationarity_Test(uv_data1['temperature_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# first difference of the columns
uv_data1['humidity_firstdiff'] = uv_data1['humidity'].diff().fillna(0.0)
uv_data1['temperature_firstdiff'] = uv_data1['temperature'].diff().fillna(0.0)

sTest = StationarityTests()
print('\nHumidity First Diff:')
sTest.ADF_Stationarity_Test(uv_data1['humidity_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature First Diff:')
sTest.ADF_Stationarity_Test(uv_data1['temperature_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))


# the Granger-Causality test
# this test tells which variable drives the other variable
# in our case, it tells us if humidity drives temperature or vice-versa
# let us take the significance level as 1% i.e., 0.01
print(grangercausalitytests(uv_data1[['humidity_firstdiff', 'temperature_firstdiff']].dropna(), 1))
print(grangercausalitytests(uv_data1[['temperature_firstdiff', 'humidity_firstdiff']].dropna(), 1))

nobs = 20
uv_data1 = uv_data1.drop(['const', 'humidity_firstdiff', 'temperature_firstdiff', 'humidity_log', 
                          'temperature_log', 'humidity_sqrt', 'temperature_sqrt'], axis=1)
main_model = uv_data1
df_train, df_test = uv_data1[0:-nobs], uv_data1[-nobs:]

plot_acf(df_train['humidity'], lags=30)
plot_pacf(df_train['humidity'])
plt.show()

plot_acf(df_train['temperature'], lags=30)
plot_pacf(df_train['temperature'])
plt.show()

model = VAR(endog=df_train)
for i in [1,2,3,4,5,6,7,8,9,10]:
    result = model.fit(i)
    print('Lag Order: ', i)
    print('HQIC: ', result.hqic)
x = model.select_order(maxlags=10)
print(x.summary())

model_fit = model.fit(26)

print('\nDurbin-Watson Test:')
testing = durbin_watson(model_fit.resid)
for col, val in zip(uv_data1.columns, testing):
    print(col, ':', round(val, 2))

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'MAPE:':mape, 'ME:':me, 'MAE:': mae, 
            'MPE:': mpe, 'RMSE:':rmse, 'Corr:':corr, 'Min-Max:':minmax})

lag_order = model_fit.k_ar
# print(lag_order)

forecast_input = df_train.values[-lag_order:]
fc = model_fit.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=uv_data1.index[-nobs:], columns=uv_data1.columns+'_forecast')
# print(df_forecast)

df_forecast['humidity_forecast'].plot(legend=True)
main_model['humidity'].plot(legend=True)
plt.show()

df_forecast['temperature_forecast'].plot(legend=True)
main_model['temperature'].plot(legend=True)
plt.show()

print('\nForecast Accuracy of Humidity:')
accuracy_prod = forecast_accuracy(df_forecast['humidity_forecast'].values, df_test['humidity'])
for k, v in accuracy_prod.items():
    print(k, round(v,4))

print('\nForecast Accuracy of Temperature:')
accuracy_prod = forecast_accuracy(df_forecast['temperature_forecast'].values, df_test['temperature'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


print('\n\nUsing ARIMA model:')
history = [x for x in df_train['humidity']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(7, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['humidity'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.humidity, label='Actual')
plt.title('Humidity')
plt.legend()
plt.show()

error = mean_squared_error(df_test['humidity'], predictions)
print('Forecast Accuracy of Humidity\nMSE: %3f' % error)

history = [x for x in df_train['temperature']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(7, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['temperature'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.temperature, label='Actual')
plt.title('Temperature')
plt.legend()
plt.show()

error = mean_squared_error(df_test['temperature'], predictions)
print('Forecast Accuracy of Temperature\nMSE: %3f' % error)




###
### DEVICE 2
###
print("\n\nDevice 2\n\n")
print(uv_data2.shape)
print(uv_data2.info()) # check the number of rows, columns and type of each column
print(uv_data2.describe()) # this gives the count, unique, top, freq entries of the data

def conv_time(x):
    x = datetime.datetime.fromtimestamp(x)
    return x

uv_data2['time'] = uv_data2['time'].apply(conv_time)
data = uv_data2
uv_data2 = data.drop(['time', 'experiment'], axis=1)
uv_data2.index = data.time
uv_data2 = uv_data2.drop(['device'], axis=1)
# print(uv_data2)

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.suptitle('Multi dimensional time series (Device 2)')
for i, ax in enumerate(axes.flatten()):
    data = uv_data2[uv_data2.columns[i]]
    ax.plot(data)
    ax.set_title(uv_data2.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()

plt.title('Device 2')
main_model = uv_data2
sns.boxplot(x=main_model['temperature'], y=main_model['humidity'])
plt.show()

humidity = uv_data2['humidity']
temperature = uv_data2['temperature']
fig, axs = plt.subplots(2)
fig.suptitle('Histogram for humidity and temperature (Device 2)')
axs[0].hist(humidity, bins=10)
axs[0].set_title('Humidity')
axs[1].hist(temperature, bins=7)
axs[1].set_title('Temperature')
plt.show()

# humidity autocorrelation plot
autocorrelation_plot(uv_data2['humidity'])
plt.title('Humidity')
plt.show()

# temperature autocorrelation plot
autocorrelation_plot(uv_data2['temperature'])
plt.title('Temperature')
plt.show()

autocorrelation_plot(uv_data2[['humidity', 'temperature']])
plt.title('Humidity and Temperature')
pyplot.show()

# correlation
print('Correlation between humidity and temperature\n')
correlation = uv_data2[['humidity', 'temperature']].corr()
print(correlation)
# we see a negative correlation of -0.40, so we can say that they are inversely proportional

# to extract maximum information from the data
# it is important to have a normal or gaussian distribution of the data
# to check fot that, I have done a normality test based on null and
# alternate hypothesis solution
print('\nHumidity:')
stat, p = stats.normaltest(uv_data2.humidity)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print('\nTemperature:')
stat, p = stats.normaltest(uv_data2.temperature)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

# visualizing the skewness and kurtosis of the data
print('\nHumidity:\n')
sns.displot(uv_data2.humidity)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data2.humidity)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data2.humidity)))

print('\nTemperature:\n')
sns.displot(uv_data2.temperature)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data2.temperature)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data2.temperature)))

# to check if each of the time-series in the system influences each other
# p-values less than significance level (0.05), implies that the hypothesis 
# X does not cause Y can be rejected
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

print('\nCausation Test:')
check_causation = grangers_causation_matrix(uv_data2, variables=uv_data2.columns)
print(check_causation)
# p value is less than significance level (0.05), so we can conclude that
# humidity and temperature are influencing each other

# to check whether or not they have unit roots by using the
# augmented dickey fuller (ADF) test
# ADF test is used to determine if the time series is stationary or not
# unit roots are a cause for the non-stationarity
sTest = StationarityTests()
print('\nHumidity:')
sTest.ADF_Stationarity_Test(uv_data2['humidity'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature:')
sTest.ADF_Stationarity_Test(uv_data2['temperature'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# from ADF Test, cannot reject null hypothesis that the series has a unit root
# so we try various methods like taking log, sqrt or difference
# to convert the series to stationary

# running on naive regression to observe what happens
# on the undifferenced time-series
uv_data2['const'] = 1
model1 = sm.OLS(endog=uv_data2['humidity'], exog=uv_data2[['temperature', 'const']])
result1 = model1.fit()
print('\n')
print(result1.summary())
plot_acf(result1.resid, alpha=0.05)
plt.show()

# transform the columns using natural log
uv_data2['humidity_log'] = np.log(uv_data2['humidity'])
uv_data2['temperature_log'] = np.log(uv_data2['temperature'])
sTest = StationarityTests()
print('\nHumidity Log:')
sTest.ADF_Stationarity_Test(uv_data2['humidity_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Log:')
sTest.ADF_Stationarity_Test(uv_data2['temperature_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# transform the columns using square root
uv_data2['humidity_sqrt'] = np.sqrt(uv_data2['humidity'])
uv_data2['temperature_sqrt'] = np.sqrt(uv_data2['temperature'])
sTest = StationarityTests()
print('\nHumidity Square Root:')
sTest.ADF_Stationarity_Test(uv_data2['humidity_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Square Root:')
sTest.ADF_Stationarity_Test(uv_data2['temperature_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# first difference of the columns
uv_data2['humidity_firstdiff'] = uv_data2['humidity'].diff().fillna(0.0)
uv_data2['temperature_firstdiff'] = uv_data2['temperature'].diff().fillna(0.0)

sTest = StationarityTests()
print('\nHumidity First Diff:')
sTest.ADF_Stationarity_Test(uv_data2['humidity_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature First Diff:')
sTest.ADF_Stationarity_Test(uv_data2['temperature_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))


# the Granger-Causality test
# this test tells which variable drives the other variable
# in our case, it tells us if humidity drives temperature or vice-versa
# let us take the significance level as 1% i.e., 0.01
print(grangercausalitytests(uv_data2[['humidity_firstdiff', 'temperature_firstdiff']].dropna(), 1))
print(grangercausalitytests(uv_data2[['temperature_firstdiff', 'humidity_firstdiff']].dropna(), 1))

nobs = 20
uv_data2 = uv_data2.drop(['const', 'humidity_firstdiff', 'temperature_firstdiff', 'humidity_log', 
                          'temperature_log', 'humidity_sqrt', 'temperature_sqrt'], axis=1)
main_model = uv_data2
df_train, df_test = uv_data2[0:-nobs], uv_data2[-nobs:]

plot_acf(df_train['humidity'], lags=30)
plot_pacf(df_train['humidity'])
plt.show()

plot_acf(df_train['temperature'], lags=30)
plot_pacf(df_train['temperature'])
plt.show()

model = VAR(endog=df_train)
for i in [1,2,3,4,5,6,7,8,9,10]:
    result = model.fit(i)
    print('Lag Order: ', i)
    print('HQIC: ', result.hqic)
x = model.select_order(maxlags=10)
print(x.summary())

model_fit = model.fit(17)

print('\nDurbin-Watson Test:')
testing = durbin_watson(model_fit.resid)
for col, val in zip(uv_data2.columns, testing):
    print(col, ':', round(val, 2))

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'MAPE:':mape, 'ME:':me, 'MAE:': mae, 
            'MPE:': mpe, 'RMSE:':rmse, 'Corr:':corr, 'Min-Max:':minmax})

lag_order = model_fit.k_ar
# print(lag_order)

forecast_input = df_train.values[-lag_order:]
fc = model_fit.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=uv_data2.index[-nobs:], columns=uv_data2.columns+'_forecast')
# print(df_forecast)

df_forecast['humidity_forecast'].plot(legend=True)
main_model['humidity'].plot(legend=True)
plt.show()

df_forecast['temperature_forecast'].plot(legend=True)
main_model['temperature'].plot(legend=True)
plt.show()

print('\nForecast Accuracy of Humidity:')
accuracy_prod = forecast_accuracy(df_forecast['humidity_forecast'].values, df_test['humidity'])
for k, v in accuracy_prod.items():
    print(k, round(v,4))

print('\nForecast Accuracy of Temperature:')
accuracy_prod = forecast_accuracy(df_forecast['temperature_forecast'].values, df_test['temperature'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


print('\n\nUsing ARIMA model:')
history = [x for x in df_train['humidity']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(7, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['humidity'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.humidity, label='Actual')
plt.title('Humidity')
plt.legend()
plt.show()

error = mean_squared_error(df_test['humidity'], predictions)
print('Forecast Accuracy of Humidity\nMSE: %3f' % error)

# different p value since they are not autocorrelated
history = [x for x in df_train['temperature']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['temperature'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.temperature, label='Actual')
plt.title('Temperature')
plt.legend()
plt.show()

error = mean_squared_error(df_test['temperature'], predictions)
print('Forecast Accuracy of Temperature\nMSE: %3f' % error)




###
### DEVICE 3
###
print("\n\nDevice 3\n\n")
print(uv_data3.shape)
print(uv_data3.info()) # check the number of rows, columns and type of each column
print(uv_data3.describe()) # this gives the count, unique, top, freq entries of the data

def conv_time(x):
    x = datetime.datetime.fromtimestamp(x)
    return x

uv_data3['time'] = uv_data3['time'].apply(conv_time)
data = uv_data3
uv_data3 = data.drop(['time', 'experiment'], axis=1)
uv_data3.index = data.time
uv_data3 = uv_data3.drop(['device'], axis=1)
# print(uv_data3)

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.suptitle('Multi dimensional time series (Device 3)')
for i, ax in enumerate(axes.flatten()):
    data = uv_data3[uv_data3.columns[i]]
    ax.plot(data)
    ax.set_title(uv_data3.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()

plt.title('Device 3')
main_model = uv_data3
sns.boxplot(x=main_model['temperature'], y=main_model['humidity'])
plt.show()

humidity = uv_data3['humidity']
temperature = uv_data3['temperature']
fig, axs = plt.subplots(2)
fig.suptitle('Histogram for humidity and temperature (Device 3)')
axs[0].hist(humidity, bins=37)
axs[0].set_title('Humidity')
axs[1].hist(temperature, bins=20)
axs[1].set_title('Temperature')
plt.show()

# humidity autocorrelation plot
autocorrelation_plot(uv_data3['humidity'])
plt.title('Humidity')
plt.show()

# temperature autocorrelation plot
autocorrelation_plot(uv_data3['temperature'])
plt.title('Temperature')
plt.show()

autocorrelation_plot(uv_data3[['humidity', 'temperature']])
plt.title('Humidity and Temperature')
pyplot.show()

# correlation
print('Correlation between humidity and temperature\n')
correlation = uv_data3[['humidity', 'temperature']].corr()
print(correlation)
# we see a negative correlation of -0.99, so we can say that they are inversely proportional

# to extract maximum information from the data
# it is important to have a normal or gaussian distribution of the data
# to check fot that, I have done a normality test based on null and
# alternate hypothesis solution
print('\nHumidity:')
stat, p = stats.normaltest(uv_data3.humidity)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print('\nTemperature:')
stat, p = stats.normaltest(uv_data3.temperature)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

# visualizing the skewness and kurtosis of the data
print('\nHumidity:\n')
sns.displot(uv_data3.humidity)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data3.humidity)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data3.humidity)))

print('\nTemperature:\n')
sns.displot(uv_data3.temperature)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data3.temperature)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data3.temperature)))

# to check if each of the time-series in the system influences each other
# p-values less than significance level (0.05), implies that the hypothesis 
# X does not cause Y can be rejected
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

check_causation = grangers_causation_matrix(uv_data3, variables=uv_data3.columns)
print(check_causation)
# p value is not less than significance level (0.05), so we can conclude that
# humidity and temperature are not influencing each other

# to check whether or not they have unit roots by using the
# augmented dickey fuller (ADF) test
# ADF test is used to determine if the time series is stationary or not
# unit roots are a cause for the non-stationarity
sTest = StationarityTests()
print('\nHumidity:')
sTest.ADF_Stationarity_Test(uv_data3['humidity'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature:')
sTest.ADF_Stationarity_Test(uv_data3['temperature'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# from ADF Test, cannot reject null hypothesis that the series has a unit root
# so we try various methods like taking log, sqrt or difference
# to convert the series to stationary

# running on naive regression to observe what happens
# on the undifferenced time-series
uv_data3['const'] = 1
model1 = sm.OLS(endog=uv_data3['humidity'], exog=uv_data3[['temperature', 'const']])
result1 = model1.fit()
print('\n')
print(result1.summary())
plot_acf(result1.resid, alpha=0.05)
plt.show()

# transform the columns using natural log -> humidity (false), temperature (true)
uv_data3['humidity_log'] = np.log(uv_data3['humidity'])
uv_data3['temperature_log'] = np.log(uv_data3['temperature'])
sTest = StationarityTests()
print('\nHumidity Log:')
sTest.ADF_Stationarity_Test(uv_data3['humidity_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Log:')
sTest.ADF_Stationarity_Test(uv_data3['temperature_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# transform the columns using square root -> humidity (false), temperature (true)
uv_data3['humidity_sqrt'] = np.sqrt(uv_data3['humidity'])
uv_data3['temperature_sqrt'] = np.sqrt(uv_data3['temperature'])
sTest = StationarityTests()
print('\nHumidity Square Root:')
sTest.ADF_Stationarity_Test(uv_data3['humidity_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Square Root:')
sTest.ADF_Stationarity_Test(uv_data3['temperature_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# first difference of the columns
uv_data3['humidity_firstdiff'] = uv_data3['humidity'].diff().fillna(0.0)
uv_data3['temperature_firstdiff'] = uv_data3['temperature'].diff().fillna(0.0)

sTest = StationarityTests()
print('\nHumidity First Diff:')
sTest.ADF_Stationarity_Test(uv_data3['humidity_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature First Diff:')
sTest.ADF_Stationarity_Test(uv_data3['temperature_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# the Granger-Causality test
# this test tells which variable drives the other variable
# in our case, it tells us if humidity drives temperature or vice-versa
# let us take the significance level as 1% i.e., 0.01
print(grangercausalitytests(uv_data3[['humidity_firstdiff', 'temperature_firstdiff']].dropna(), 1))
print(grangercausalitytests(uv_data3[['temperature_firstdiff', 'humidity_firstdiff']].dropna(), 1))

nobs = 20
uv_data3 = uv_data3.drop(['const', 'humidity_firstdiff', 'temperature_firstdiff', 'humidity_log', 
                          'temperature_log', 'humidity_sqrt', 'temperature_sqrt'], axis=1)
main_model = uv_data3
df_train, df_test = uv_data3[0:-nobs], uv_data3[-nobs:]

plot_acf(df_train['humidity'], lags=50)
plot_pacf(df_train['humidity'])
plt.show()

plot_acf(df_train['temperature'], lags=50)
plot_pacf(df_train['temperature'])
plt.show()

model = VAR(endog=df_train)
for i in [1,2,3,4,5,6,7,8,9,10]:
    result = model.fit(i)
    print('Lag Order: ', i)
    print('HQIC: ', result.hqic)
x = model.select_order(maxlags=10)
print(x.summary())

model_fit = model.fit(47)

testing = durbin_watson(model_fit.resid)
for col, val in zip(uv_data3.columns, testing):
    print(col, ':', round(val, 2))

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'MAPE:':mape, 'ME:':me, 'MAE:': mae, 
            'MPE:': mpe, 'RMSE:':rmse, 'Corr:':corr, 'Min-Max:':minmax})

lag_order = model_fit.k_ar
# print(lag_order)

forecast_input = df_train.values[-lag_order:]
fc = model_fit.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=uv_data3.index[-nobs:], columns=uv_data3.columns+'_forecast')
# print(df_forecast)

df_forecast['humidity_forecast'].plot(legend=True)
main_model['humidity'].plot(legend=True)
plt.show()

df_forecast['temperature_forecast'].plot(legend=True)
main_model['temperature'].plot(legend=True)
plt.show()

print('\nForecast Accuracy of Humidity:')
accuracy_prod = forecast_accuracy(df_forecast['humidity_forecast'].values, df_test['humidity'])
for k, v in accuracy_prod.items():
    print(k, round(v,4))

print('\nForecast Accuracy of Temperature:')
accuracy_prod = forecast_accuracy(df_forecast['temperature_forecast'].values, df_test['temperature'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


print('\n\nUsing ARIMA model:')
history = [x for x in df_train['humidity']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['humidity'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.humidity, label='Actual')
plt.title('Humidity')
plt.legend()
plt.show()

error = mean_squared_error(df_test['humidity'], predictions)
print('Forecast Accuracy of Humidity\nMSE: %3f' % error)

history = [x for x in df_train['temperature']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['temperature'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.temperature, label='Actual')
plt.title('Temperature')
plt.legend()
plt.show()

error = mean_squared_error(df_test['temperature'], predictions)
print('Forecast Accuracy of Temperature\nMSE: %3f' % error)




###
### DEVICE 4
###
print("\n\nDevice 4\n\n")
print(uv_data4.shape)
print(uv_data4.info()) # check the number of rows, columns and type of each column
print(uv_data4.describe()) # this gives the count, unique, top, freq entries of the data

# as observed in describe(), there are 'None' values in humidity and temperature
# data cleaning
uv_data4['humidity'] = uv_data4.humidity.replace('None', 0, regex=True).astype(float)
uv_data4['temperature'] = uv_data4.temperature.replace('None', 0, regex=True).astype(float)

columns = ['attribute', '# of zeros']
zeros = (uv_data4 == 0).sum(axis=0)
count = pd.DataFrame(columns=columns)
count.loc[0] = ['device'] +  [zeros[0]]
count.loc[1] = ['humidity'] +  [zeros[1]]
count.loc[2] = ['temperature'] +  [zeros[2]]
count.loc[3] = ['experiment'] +  [zeros[3]]
count.loc[4] = ['time'] +  [zeros[4]]
print('\n')
print(count)
count.plot.bar(x='attribute', y='# of zeros')
plt.show()

print(uv_data4.info())
print(uv_data4.describe())

# from the above describe() function, we get mean as 36.61 for humidity and 24.22 for temperature
# replace the 0's with the mean
uv_data4['humidity'] = uv_data4.humidity.replace(0, 36.61)
uv_data4['temperature'] = uv_data4.temperature.replace(0, 24.22)

print(uv_data4.info())
print(uv_data4.describe())

def conv_time(x):
    x = datetime.datetime.fromtimestamp(x)
    return x

uv_data4['time'] = uv_data4['time'].apply(conv_time)
data = uv_data4
uv_data4 = data.drop(['time', 'experiment'], axis=1)
uv_data4.index = data.time
uv_data4 = uv_data4.drop(['device'], axis=1)
# print(uv_data4)

fig, axes = plt.subplots(nrows=2, ncols=1)
fig.suptitle('Multi dimensional time series (Device 4)')
for i, ax in enumerate(axes.flatten()):
    data = uv_data4[uv_data4.columns[i]]
    ax.plot(data)
    ax.set_title(uv_data4.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()

main_model = uv_data4
plt.title('Device 4')
sns.boxplot(x=main_model['temperature'], y=main_model['humidity'])
plt.show()

humidity = uv_data4['humidity']
temperature = uv_data4['temperature']
fig, axs = plt.subplots(2)
fig.suptitle('Histogram for humidity and temperature (Device 4)')
axs[0].hist(humidity, bins=28)
axs[0].set_title('Humidity')
axs[1].hist(temperature, bins=17)
axs[1].set_title('Temperature')
plt.show()

# humidity autocorrelation plot
autocorrelation_plot(uv_data4['humidity'])
plt.title('Humidity')
plt.show()

# temperature autocorrelation plot
autocorrelation_plot(uv_data4['temperature'])
plt.title('Temperature')
plt.show()

autocorrelation_plot(uv_data4[['humidity', 'temperature']])
plt.title('Humidity and Temperature')
pyplot.show()

# correlation
print('Correlation between humidity and temperature\n')
correlation = uv_data4[['humidity', 'temperature']].corr()
print(correlation)
# we see a negative correlation of -0.71, so we can say that they are inversely proportional

# to extract maximum information from the data
# it is important to have a normal or gaussian distribution of the data
# to check fot that, I have done a normality test based on null and
# alternate hypothesis solution
print('\nHumidity:')
stat, p = stats.normaltest(uv_data4.humidity)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

print('\nTemperature:')
stat, p = stats.normaltest(uv_data4.temperature)
print('Statistics=%.3f, p=%.3f' %(stat, p))
alpha = 0.05
if p > alpha:
    print('Data looks Gaussian (fail to reject H0)')
else:
    print('Data does not look Gaussian (reject H0)')

# visualizing the skewness and kurtosis of the data
print('\nHumidity:\n')
sns.displot(uv_data4.humidity)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data4.humidity)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data4.humidity)))

print('\nTemperature:\n')
sns.displot(uv_data4.temperature)
print('Kurtosis of normal distribution: {}'.format(stats.kurtosis(uv_data4.temperature)))
print('Skewness of normal distribution: {}'.format(stats.skew(uv_data4.temperature)))

# to check if each of the time-series in the system influences each other
# p-values less than significance level (0.05), implies that the hypothesis 
# X does not cause Y can be rejected
maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

check_causation = grangers_causation_matrix(uv_data4, variables=uv_data4.columns)
print(check_causation)
# p value is less than significance level (0.05), so we can conclude that
# humidity and temperature are influencing each other

# to check whether or not they have unit roots by using the
# augmented dickey fuller (ADF) test
# ADF test is used to determine if the time series is stationary or not
# unit roots are a cause for the non-stationarity
sTest = StationarityTests()
print('\nHumidity:')
sTest.ADF_Stationarity_Test(uv_data4['humidity'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature:')
sTest.ADF_Stationarity_Test(uv_data4['temperature'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# from ADF Test, cannot reject null hypothesis that the series has a unit root
# so we try various methods like taking log, sqrt or difference
# to convert the series to stationary

# running on naive regression to observe what happens
# on the undifferenced time-series
uv_data4['const'] = 1
model1 = sm.OLS(endog=uv_data4['humidity'], exog=uv_data4[['temperature', 'const']])
result1 = model1.fit()
print('\n')
print(result1.summary())
plot_acf(result1.resid, alpha=0.05)
plt.show()

# transform the columns using natural log
uv_data4['humidity_log'] = np.log(uv_data4['humidity'])
uv_data4['temperature_log'] = np.log(uv_data4['temperature'])
sTest = StationarityTests()
print('\nHumidity Log:')
sTest.ADF_Stationarity_Test(uv_data4['humidity_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Log:')
sTest.ADF_Stationarity_Test(uv_data4['temperature_log'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# transform the columns using square root
uv_data4['humidity_sqrt'] = np.sqrt(uv_data4['humidity'])
uv_data4['temperature_sqrt'] = np.sqrt(uv_data4['temperature'])
sTest = StationarityTests()
print('\nHumidity Square Root:')
sTest.ADF_Stationarity_Test(uv_data4['humidity_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature Square Root:')
sTest.ADF_Stationarity_Test(uv_data4['temperature_sqrt'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# first difference of the columns
uv_data4['humidity_firstdiff'] = uv_data4['humidity'].diff().fillna(0.0)
uv_data4['temperature_firstdiff'] = uv_data4['temperature'].diff().fillna(0.0)

sTest = StationarityTests()
print('\nHumidity First Diff:')
sTest.ADF_Stationarity_Test(uv_data4['humidity_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))
print('\nTemperature First Diff:')
sTest.ADF_Stationarity_Test(uv_data4['temperature_firstdiff'], printResults=True)
print('Is the time series stationary? {0}'.format(sTest.isStationary))

# the Granger-Causality test
# this test tells which variable drives the other variable
# in our case, it tells us if humidity drives temperature or vice-versa
# let us take the significance level as 1% i.e., 0.01
print(grangercausalitytests(uv_data4[['humidity_firstdiff', 'temperature_firstdiff']].dropna(), 1))
print(grangercausalitytests(uv_data4[['temperature_firstdiff', 'humidity_firstdiff']].dropna(), 1))

nobs = 20
uv_data4 = uv_data4.drop(['const', 'humidity_firstdiff', 'temperature_firstdiff', 'humidity_log', 
                          'temperature_log', 'humidity_sqrt', 'temperature_sqrt'], axis=1)
main_model = uv_data4
df_train, df_test = uv_data4[0:-nobs], uv_data4[-nobs:]

plot_acf(df_train['humidity'], lags=40)
plot_pacf(df_train['humidity'])
plt.show()

plot_acf(df_train['temperature'], lags=40)
plot_pacf(df_train['temperature'])
plt.show()

model = VAR(endog=df_train)
for i in [1,2,3,4,5,6,7,8,9,10]:
    print('Lag Order: ', i)
    print('HQIC: ', result.hqic)
x = model.select_order(maxlags=10)
print(x.summary())

model_fit = model.fit(34)

testing = durbin_watson(model_fit.resid)
for col, val in zip(uv_data4.columns, testing):
    print(col, ':', round(val, 2))

def forecast_accuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'MAPE:':mape, 'ME:':me, 'MAE:': mae, 
            'MPE:': mpe, 'RMSE:':rmse, 'Corr:':corr, 'Min-Max:':minmax})

lag_order = model_fit.k_ar
# print(lag_order)

forecast_input = df_train.values[-lag_order:]
fc = model_fit.forecast(y=forecast_input, steps=nobs)
df_forecast = pd.DataFrame(fc, index=uv_data4.index[-nobs:], columns=uv_data4.columns+'_forecast')
# print(df_forecast)

df_forecast['humidity_forecast'].plot(legend=True)
main_model['humidity'].plot(legend=True)
plt.show()

df_forecast['temperature_forecast'].plot(legend=True)
main_model['temperature'].plot(legend=True)
plt.show()

print('\nForecast Accuracy of Humidity:')
accuracy_prod = forecast_accuracy(df_forecast['humidity_forecast'].values, df_test['humidity'])
for k, v in accuracy_prod.items():
    print(k, round(v,4))

print('\nForecast Accuracy of Temperature:')
accuracy_prod = forecast_accuracy(df_forecast['temperature_forecast'].values, df_test['temperature'])
for k, v in accuracy_prod.items():
    print(k, ': ', round(v,4))


print('\n\nUsing ARIMA model:')
history = [x for x in df_train['humidity']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(6, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['humidity'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.humidity, label='Actual')
plt.title('Humidity')
plt.legend()
plt.show()

error = mean_squared_error(df_test['humidity'], predictions)
print('Forecast Accuracy of Humidity\nMSE: %3f' % error)

history = [x for x in df_train['temperature']]
predictions = list()
values = list()
for t in range(len(df_test)):
    model = ARIMA(history, order=(7, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    values.append(yhat[0])
    predictions.append(yhat)
    obs = df_test['temperature'][t]
    history.append(obs)
    print('predicted:%f, expected:%f' %(yhat, obs))

plt.plot(df_test.index, values, label='Forecast', color='red')
plt.plot(df_test.index, df_test.temperature, label='Actual')
plt.title('Temperature')
plt.legend()
plt.show()

error = mean_squared_error(df_test['temperature'], predictions)
print('Forecast Accuracy of Temperature\nMSE: %3f' % error)