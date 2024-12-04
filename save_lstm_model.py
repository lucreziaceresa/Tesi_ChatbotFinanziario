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


def create_sequence(dataset, label_columns, step_size):
    sequences = []
    labels = []
    start_idx = 0

    for stop_idx in range(step_size,len(dataset)): 
        sequences.append(dataset.iloc[start_idx:stop_idx,:])
        labels.append(dataset.iloc[stop_idx,:][label_columns])
        start_idx += 1
    return (np.array(sequences),np.array(labels))

step_size = 20
train_seq, train_label = create_sequence(dataset.iloc[:-20,:], columns_scaled, step_size)

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
    
input_size = len(dataset.columns)
lstm_units = [50, 100]
num_layers = 1
dense_units = [32, 64]
output_size = 2
learning_rates = [0.001, 0.01, 0.1]

criterion = nn.MSELoss()

best_loss = float('inf')

for fold in range(train_seq.shape[0] - 1):
    for lstm_unit in lstm_units:
        for dense_unit in dense_units:
            for learning_rate in learning_rates:

                model = LSTM_network(input_size=input_size, hidden_size=lstm_unit, num_layers=num_layers, 
                                     output_size=output_size, dense_size=dense_unit)
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                train_seq_fold = torch.from_numpy(train_seq[:fold+1]).float()
                train_label_fold = torch.from_numpy(train_label[:fold+1]).float()

                val_seq_fold = torch.from_numpy(train_seq[fold+1]).float()
                val_label_fold = torch.from_numpy(train_label[fold+1]).float()

                patience = 0
                while patience < 10:
                    model.train()
                    out = model(train_seq_fold)
                    loss = criterion(torch.stack([o[-1] for o in out]), train_label_fold)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    model.eval()
                    with torch.no_grad():
                        out_val = model(val_seq_fold)
                        loss_val = criterion(out_val[-1], val_label_fold)
                        
                        print(f'Fold {fold}:    Val Loss = {round(float(loss_val), 6)} | Best Loss = {round(float(best_loss), 6)}', end="\r")

                        if loss_val < best_loss:
                            best_loss = loss_val
                            best_model = {
                                'model': model,
                                'lstm_unit': lstm_unit,
                                'dense_unit': dense_unit,
                                'input_size': input_size,
                                'num_layers': num_layers,
                                'output_size': output_size
                            }
                            patience = 0
                        else:
                            patience += 1


torch.save({
    'model_state_dict': best_model['model'].state_dict(),
    'lstm_unit': best_model['lstm_unit'],
    'dense_unit': best_model['dense_unit'],
    'input_size': best_model['input_size'],
    'num_layers': best_model['num_layers'],
    'output_size': best_model['output_size'],
    'best_loss': best_loss
}, 'best_model.pth')