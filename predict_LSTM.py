import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import mysql.connector


### IMPORT UPDATED DATASET ###

format = '%Y-%m-%d %H:%M'

finbert_aapl_path = Path('..').resolve() / 'Updated-datasets' / 'finbert_aapl.csv'
finbert_aapl = pd.read_csv(finbert_aapl_path)
finbert_aapl['datetime'] = [datetime.strptime(d, format) for d in finbert_aapl['datetime']]

prices_path = Path('..').resolve() / 'Updated-datasets' / 'prices.csv'
prices = pd.read_csv(prices_path)
prices = prices[prices['symbol'] == 'AAPL']
prices['datetime'] = [datetime.strptime(d, format) for d in prices['datetime']]

df = pd.merge(finbert_aapl, prices, 'outer', 'datetime')
df = df.sort_values(by='datetime', ascending=True)

finbert_columns = [c for c in df.columns if 'norm' in c]

dates = sorted(df[df['price'].isna() == False]['datetime'].to_list())
idx_dates = df[df['price'].isna() == False].index
start_date = min(dates) - timedelta(days=2)

for i, (date, idx) in enumerate(zip(dates, idx_dates)):
    if len(df[(df['datetime'] <= date) & (df['datetime'] > start_date)]) > 1:
        mean = df[(df['datetime'] <= date) & (df['datetime'] > start_date)][finbert_columns].mean()
        df.loc[idx,finbert_columns] = mean

    else:
        if i == 0:
            df.loc[idx,finbert_columns] = (0,0,1)
        else:
            if df.loc[idx,finbert_columns].isna().sum() == 3:
                df.loc[idx,finbert_columns] = df.loc[idx_dates[i-1],finbert_columns]
    
    start_date = date

df_for_lstm = df[df['price'].isna() == False][['datetime'] + finbert_columns + ['price']]





### LSTM ###

dataset = df_for_lstm[finbert_columns + ['price']]
dataset.index = df_for_lstm['datetime']
dataset['price_in5min'] = list(dataset['price'][1:].values) + [None]
dataset = dataset.iloc[:-1,:]

if len(dataset) > 500:
    dataset = dataset.iloc[-500:,:]

 
mms = MinMaxScaler()
columns_scaled = ['price_scaled', 'price_in5min_scaled']
columns_to_scale = ['price', 'price_in5min']
dataset[columns_scaled] = mms.fit_transform(dataset[columns_to_scale])
dataset = dataset[finbert_columns + columns_scaled]

date = dataset.index[-1] + timedelta(minutes=5)
date = datetime.strftime(date, format)
test = np.array(dataset.iloc[-20:,:])




checkpoint = torch.load('best_model.pth')




class LSTM_network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dense_size):
        super(LSTM_network, self).__init__()
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.dense1 = nn.Linear(hidden_size, dense_size)
        # self.dense2 = nn.Linear(dense_size, dense_size)
        self.output = nn.Linear(dense_size, output_size)
    
    def forward(self, X_train):
        out_lstm, (hn, cn) = self.lstm(X_train)
        out_lstm_relu = self.relu(out_lstm)
        out_dense1 = self.dense1(out_lstm_relu)
        out_dense1_relu = self.relu(out_dense1)
        output = self.output(out_dense1_relu)

        return output


model = LSTM_network(
    input_size=checkpoint['input_size'],
    hidden_size=checkpoint['lstm_unit'],
    num_layers=checkpoint['num_layers'],  
    output_size=checkpoint['output_size'],
    dense_size=checkpoint['dense_unit']
)

model.load_state_dict(checkpoint['model_state_dict'])


model.eval()
with torch.no_grad():
    prediction_scaled = model(torch.from_numpy(test).float())

prediction = mms.inverse_transform([[p[0], p[1]] for p in prediction_scaled])


db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="chatbot"
)
cursor = db.cursor()


cursor.execute('INSERT INTO prediction (datetim, price, price_in5min) VALUES (%s,%s,%s)', tuple(date, prediction[0], prediction[1]))
db.commit()

cursor.close()
db.close()