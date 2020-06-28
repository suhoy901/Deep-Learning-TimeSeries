# -*- coding: utf-8 -*-
"""
time_series.ipynb
(1)forecast a univariate time series
(2)multivariate time series.
"""

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

"""## The weather dataset"""

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)

"""Let's take a glance at the data."""

df.head()

df.shape


### Single Step Prediction(history size만 설정)
"""*매 10분마다 측정해서 1시간에 6개의 observation이 있으며 하루에 144개의 데이터가 있음 

*특정시점에서 6시간뒤의 온도를 예측하는 모형을 생성 
  5일간의 데이터를 사용-> 144*5 = 720개의 데이터를 사용하여 학습. 

*하루에 144개

*'history_size'는 과거의 정보 'target_size'는 얼마나 먼 미래를 예측하고 싶은지 우리가 원하는예측값의 사이즈
"""

def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
    return np.array(data), np.array(labels)

"""*학습데이터: 300,000 rows(~2100일) 나머지는 확인 데이터."""

TRAIN_SPLIT = 300000 # 학습데이터셋으로 활용할 갯수

"""Setting seed to ensure reproducibility."""

tf.random.set_seed(13)


##### 일변량 예측 분석 : 독립변수 1개, 종속변수 1개
"""## Part 1: Forecast a univariate time series
*self데이터만을 사용해서 예측.
"""

uni_data = df['T (degC)']
uni_data.index = df['Date Time']
uni_data.head()

"""Let's observe how this data looks across time."""

uni_data.plot(subplots=True)

uni_data = uni_data.values
uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = (uni_data-uni_train_mean)/uni_train_std

univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

def create_time_steps(length):
    return list(range(-length, 0))

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize=10, label=labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future+5)*2])
    plt.xlabel('Time-Step')
    return plt

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

"""### Baseline
베이스라인 --> 과거 20개의 observation의 평균을 예측값으로 설정
"""

def baseline(history):
    return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0, 'Baseline Prediction Example')


BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()


### LSTM모형으로 
"""LSTM모형 정의"""
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')


for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)


for x, y in val_univariate.take(3):
    plot = show_plot([x[0].numpy(), y[0].numpy(), simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
    plot.show()



### 독립변수 : Multivariate설정(3가지), 종속변수 1개
## 1. Single step prediction로 접근하기
## 2. Multistep prediction
"""
## Part 2: Forecast a multivariate time series
원 데이터에는 14개의 feature가 포함되어있으나 예쩨에서는 3개만 포함. (온도, 기압, air density)
"""

features_considered = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']

features = df[features_considered]
features.index = df['Date Time']
features.head()


features.plot(subplots=True)
print(dataset.shape)

# timestamp상에서 선행적으로 norm
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

"""### Single step model
Single Step에서는 과거를 통해 한개의 포인트를 예측 

위의 예제와 같은 윈도우를 취함.
"""

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step) # step을 사용하는 이유 : 소량의 데이터도 충분히 패턴을 대표할 수 있다는 관점임
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

"""
*5일 720개
*하루 144개
*한시간 6개

네트워크는 지난 5일 (720개의 값) 의 데이터가 매 시간 측정됨. 이 예제에서는 1시간동안 큰 변화가 없다는 가정하에 시간당의 샘플링을 취함. 
따라서 지난 720개가 아닌 120개의 데이터가 지난 5일간의 과거를 표현. 
Single Step Prediction a모형에서는 12시간 뒤의 데이터포인트였으나 이번에는 72개 뒤의 값을 책정 (12*6)
"""

past_history = 720 # 하루 144개 * 5 = 720(과거 5일간의 데이터를 기반으로 72시간 뒤의 값을 예측하는 모델)
future_target = 72 # 특정 타임스탬프를 기준으로 얼마나 뒤의 값을 예측할 것인지(72 -> 72시간 이후)
STEP = 6


x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)


print ('Single window of past history : {}'.format(x_train_single[0].shape))

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

"""Let's check out a sample prediction."""

for x, y in val_data_single.take(1):
    print(single_step_model.predict(x).shape)

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()

plot_train_history(single_step_history, 'Single Step Training and validation loss')

"""#### Predict a single step future
모형이 학습이되었으니 예측값을 생성해보자. 
3개의 feature의 5일간의 시간당 과거값을 사용 (120개의 데이터들)하여 온도값을 예측. 
예측값은 하루뒤의 값을 예측하도록 설정
"""

for x, y in val_data_single.take(3):
    plot = show_plot([x[0][:, 1].numpy(), y[0].numpy(), single_step_model.predict(x)[0]], 12, 'Single Step Prediction')
    plot.show()

## Multi-step prediction
"""### Multi-Step model
멀티스텝 모형에서는 과거가 주어졌을때 미래 값의 Range를 예측하도록 설정. (Sequence를 예측)

멀티스텝모형에서 역시 학습 데이터는 과거 5일간의 시간당 샘플된 데이터를 사용. 
이 모형에서는 다음 12시간동안의 온도를 예측 (매 10분마다 측정된 값이므로 72개의 예측값을 뱉어내야 함)
"""

future_target = 72
x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 1], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 1],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

print ('Single window of past history : {}'.format(x_train_multi[0].shape))
print ('\n Target temperature to predict : {}'.format(y_train_multi[0].shape))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()


"""Plotting a sample data-point."""

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo', label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro', label='Predicted Future')
    plt.legend(loc='upper left')
    plt.show()



for x, y in train_data_multi.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))



multi_step_model = tf.keras.models.Sequential()
multi_step_model.add(tf.keras.layers.LSTM(32,
                                          return_sequences=True,
                                          input_shape=x_train_multi.shape[-2:]))
multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
multi_step_model.add(tf.keras.layers.Dense(72))

multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mae')

"""Let's see how the model predicts before it trains."""

for x, y in val_data_multi.take(1):
    print (multi_step_model.predict(x).shape)

multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

plot_train_history(multi_step_history, 'Multi-Step Training and validation loss')

"""#### Predict a multi-step future
Let's now have a look at how well your network has learnt to predict the future.
"""

for x, y in val_data_multi.take(3):
    multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])