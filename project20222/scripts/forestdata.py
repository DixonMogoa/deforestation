import datetime
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from dateutil.relativedelta import relativedelta
#for normalizing data
from sklearn.preprocessing import MinMaxScaler

def get_data() :
    scaler = MinMaxScaler(feature_range=(0, 1))

    df = pd.read_csv("forest.csv")

    #setting index as date
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = df['Date']

    #creating dataframe
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'hectare'])
    for i in range(0,len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['hectare'][i] = data['hectare'][i]

    #setting index
    new_data.index = new_data.Date

    new_data.drop('Date', axis=1, inplace=True)

    #creating train and test sets
    dataset = new_data.values

    train = dataset[0:5,:]
    valid = dataset[4,:]

    #converting dataset into x_train and y_train
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(4,len(train)):
        x_train.append(scaled_data[i-4:i,0])
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

    #predicting 11 values, using past 4 from the train data
    inputs = new_data[len(new_data) - len(valid) - 4:].values
    inputs = inputs.reshape(-1,1)
    inputs  = scaler.transform(inputs)

    X_test = []
    for i in range(4,inputs.shape[0]):
        X_test.append(inputs[i-4:i,0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))

    # generate data for 100 years as from 2013 data
    X_FUTURE = 11
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
      curr_date = curr_date + relativedelta(years=10)
      dicts.append({"Date": curr_date, 'hectare':predictions[i]})

    new_data = pd.DataFrame(dicts)
    new_csv_data = df.append(new_data)

    return new_csv_data.to_json(orient='records')
    