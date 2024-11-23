# Chapter 9 ARIMA models
ARIMA models provide another approach to time series forecasting. Exponential smoothing and ARIMA models are the two most widely used approaches to time series forecasting, and provide complementary approaches to the problem. While exponential smoothing models are based on a description of the trend and seasonality in the data, ARIMA models aim to describe the autocorrelations in the data.

ARIMA stands for **Autoregressive Integrated Moving Average Model**. It belongs to a class of models that explains a given time series based on its own past values. i.e., its own lags and the lagged forecast errors. The equation can be used to forecast future values. Any 'non-seasonal' time series that exhibits patterns and is not a random white noise can be modelled with ARIMA models.

ARIMA models are specified by three order parameters: $(p, d, q)$, where
- $p$ is the order of the Autoregressive (AR) term
- $q$ is the order of the Moving Average (MA) term
- $d$ is the number of differencing required to make the time series stationary.

### AR and MA Models as components of ARIMA
- **AR(p) Autoregression** - a regression model that utilises the dependent relationship between a current observation and observation over a previous period. An autoregressive (AR(p)) component refers to the use of past values in the regression equation for the time series. $p$ is the number of lags of $Y$ to be used as predictors.

- **I(d) Integration** - uses differencing of observations (subtracting an observation from observation at the previous timestamp) in order to make the time series stationary. Differencing involves involves the subtraction of the current values of a series with its previous values $d$ number of times. The AR in ARIMA means it is a linear regression model that uses its own lag as predictors. Linear regression models work best when the predictors are not correlated and are independent of each other, which is why we need to make the time series stationary. The value of $d$ is the minimum number of differencing needed to make the time series stationary. If the time series is already stationary, then $d=0$. 

- **MA(q) Moving Average** - A model that uses the dependency between an observation and a residual error from a moving average model applied to lagged observations. The moving average components depicts the error of the model as a combination of previous error terms. The order $q$ represents the number of terms to be included in the model. 

### Types of ARIMA Model
- **ARIMA**: Non-seasonal Autoregressive Integrated Moving Averages
- **SARIMA**: Seasonal ARIMA (when the time series has seasonal patterns)
- **SARIMAX**: Seasonal ARIMA with exogenous variables

#### ARIMA model in words
Predicted $y_t$ equals a constant plus linear combinations of the lags of $Y$ (up to $p$ lags) plus linear combination of lagged forecast errors (up to $q$ lags).