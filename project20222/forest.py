import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta
#setting figure size
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10

#for normalizing data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

style.use('ggplot')

df = pd.read_csv("forest1.csv")

#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%Y-%m-%d')
df.index = df['Date']

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Value'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Value'][i] = data['Value'][i]

#setting index
new_data.index = new_data.Date

new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

train = dataset[0:987,:]
valid = dataset[987:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

#predicting 10 values, using past 3 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

# generate data for 100 years as from 2013 data
X_FUTURE = 100
predictions = np.array([])
last = x_train[-1]
for i in range(X_FUTURE):
  curr_prediction = model.predict(np.array([last]))
  last = np.concatenate([last[1:], curr_prediction])
  predictions = np.concatenate([predictions, curr_prediction[0]])
predictions = scaler.inverse_transform([predictions])[0]

dicts = []
curr_date = data.index[-1]
for i in range(X_FUTURE):
  curr_date = curr_date + relativedelta(days=1)
  dicts.append({'Value':predictions[i], "Date": curr_date})

new_data = pd.DataFrame(dicts).set_index("Date")
new_csv_data = df.append(new_data)

#Plot the data
# train = df
train = df[:987]
new_data = new_data[987:]
#Visualize the data
plt.figure(figsize=(16,8))
plt.title('Deforestation of Mau Forest Over Years')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Hectare', fontsize=18)
plt.plot(train['Value'])
plt.plot(new_data['Value'])
plt.legend(['Train', 'Predictions'], loc='lower right')
plt.show()
