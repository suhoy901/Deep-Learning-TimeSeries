import pandas as pd
import numpy as np
from fbprophet import Prophet
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
import plotly.io as pio

sns.set_style("whitegrid")
#pio.renderers.default = "browser"
warnings.filterwarnings('ignore')

train = pd.read_csv(
    'C:/Users/yckim/workspace/TS/TS/1/dataset/Train_SU63ISt.csv')
test = pd.read_csv(
    'C:/Users/yckim/workspace/TS/TS/1/dataset/Test_0qrQsBZ.csv')


### 1. Datetime column : 'object' type -> 'datetime64' type
train['Datetime'] = pd.to_datetime(train["Datetime"], format='%d-%m-%Y %H:%M')
test['Datetime'] = pd.to_datetime(test["Datetime"], format='%d-%m-%Y %H:%M')
# print("train 데이터의 shape: ", train.shape) # (18288, 3)
# print("test 데이터의 shape:", test.shape)    # (5112, 2)


### 2. Add new column 'hour' to train dataframe
train['hour'] = train['Datetime'].dt.hour


### 3. Calculate average hourly fraction
hourly_frac = train.groupby(['hour']).mean() / \
    np.sum(train.groupby(['hour']).mean())
hourly_frac.drop(['ID'], axis=1, inplace=True)
hourly_frac.columns = ['fraction']

### 4. index를 datetime으로 변경
train.index = train.Datetime

### 5. drop columns
train.drop(["ID", "hour", "Datetime"], axis=1, inplace=True)

### 6. Visualize train data : hourly 
plt.figure(figsize=(20, 10))
plt.xlabel("Time(Year-Month)")
plt.ylabel("Passenger count")
plt.plot(train, label="Passenger Count")
plt.show()

"""
trace1 = go.Scatter(
    x=train.index,
    y=train.Count,
    name="Passenger Count",
    line=dict(color="blue"),
    opacity=0.4)

layout1 = dict(title="Passenger Count",)

fig = dict(data=[trace1], layout=layout1)
iplot(fig)
"""


### 7. Visualize train data : daily
daily_train = train.resample('D').sum()

plt.figure(figsize=(20, 10))
plt.xlabel("Time(Year-Month")
plt.ylabel("Passenger Count")
plt.plot(daily_train, label="Passenger Count")
plt.show()
# sns.lineplot(x=daily_train.index, y="Count", data=daily_train)

"""
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_train.index,
    y=daily_train.Count,
    name="Passenger Count",
    line=dict(color="blue"),
    opacity=0.4))

fig.update_layout(title="Passenger Count", xaxis_title='Time(Year-Month)', yaxis_title='Count')
fig.show()
"""

### 7. Prophet
daily_train['ds'] = daily_train.index
daily_train['y'] = daily_train.Count
daily_train.drop(['Count'], axis=1, inplace=True)


### 8. Prophet fitting
m_passenger = Prophet(yearly_seasonality=True, seasonality_prior_scale=0.3)
m_passenger.fit(daily_train)
future_dataframe = m_passenger.make_future_dataframe(periods=213)
forecast = m_passenger.predict(future_dataframe)

fig = m_passenger.plot_components(forecast)
plt.show()

### 9. Extract hour, day, month and year from both dataframes to merge
forecast = forecast.rename(columns={'ds': 'Datetime'})

for df in [test, forecast]:
    df['hour'] = df['Datetime'].dt.hour
    df['day'] = df['Datetime'].dt.day
    df['month'] = df['Datetime'].dt.month
    df['year'] = df['Datetime'].dt.year

test = pd.merge(test, forecast, on=['hour', 'day', 'month', 'year'], how='left')
cols = ['ID', 'hour', 'yhat']
test_new = test[cols]

#######################
test_new = pd.merge(test_new, hourly_frac, left_on=['hour'], right_index=True, how='left')
test_new.fillna(method='ffill', inplace=True)

test_new['Count'] = test_new['yhat'] * test_new['fraction']
test_new.drop(['yhat', 'fraction', 'hour'], axis=1, inplace=True)

test_new.to_csv('C:/Users/yckim/workspace/TS/TS/1/prophet_submmistion.csv', index=False)