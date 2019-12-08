import csv
import pandas as pd
import numpy as np
import re
import math
import timeit
from sklearn.metrics import mean_squared_error


# In[2]:

def get_lagged_features(value, n_steps):
    lag_list = []
    for lag in range(n_steps, 0, -1):
        lag_list.append(value.shift(lag))
    return pd.concat(lag_list, axis=1)


# In[3]:

def add_lagged_returns(df,ret,xs_ret):
  lags = 3 # 3-months returns have already been observed at t-3 
  df_lagged = df
  retL = ret + '.L' + str(lags)
  xs_retL = xs_ret + '.L' + str(lags)
  df_lagged[retL] = df.groupby('SecurityID').transform(lambda x: x.shift(-lags))[ret]
  df_lagged[xs_retL] = df.groupby('SecurityID').transform(lambda x: x.shift(-lags))[xs_ret]
  df_lagged.dropna(axis=0, how='any', subset=[retL,xs_retL], inplace=True)
  return df_lagged



def load_data(infile, ret, xs_ret):
  
  # Load the book data
  dateparse = lambda x: pd.datetime.strptime(x, '%Y%m')
  df = pd.read_csv(infile, parse_dates=['eom'], date_parser=dateparse)


  # Clean the data
  # Report the percentage of missing values per column

  na_pcts = (df.isnull().sum()/df.shape[0])*100
  print na_pcts

  # Dropping columns with a great percentage of missing values than threshold
  threshold = 10
  drop_cols = []
  for key,value in na_pcts.iteritems():
    if (na_pcts[key]>=threshold):
       drop_cols.append(key)

  df.drop(drop_cols,axis=1,inplace=True)
  df.dropna(axis=0, how='any', subset=['SectorID'], inplace=True)


  df['Sector']=df['SectorID'].apply(lambda x: int(str(x)[:2]))
  df['IndGroup']=df['SectorID'].apply(lambda x: int(str(x)[2:4]))
  df['Industry']=df['SectorID'].apply(lambda x: int(str(x)[4:6]))
  df['SubIndustry']=df['SectorID'].apply(lambda x: int(str(x)[6:8]))
  
  # fill-in missing values by using industry group medians
  df_filled=df.groupby(['Sector','IndGroup','eom']).transform(lambda x: x.fillna(x.median()))
  df_filled[['Sector','IndGroup','eom']]=df[['Sector','IndGroup','eom']]
  df_sorted=df_filled.sort_values(['SecurityID', 'eom'], ascending=[True, False])
  col_names = list(df_sorted.columns.values)
  unwanted = {'SecurityID','SectorID','Sector','IndGroup','Industry','SubIndustry','eom'}
  col_names = [e for e in col_names if e not in unwanted]

  return(df_sorted)

# serialize models to files (for reproducability)
def serialize_models(symbols, models, x_trains, x_tests, y_trains, y_tests):

  for symbol in symbols:
    symbol
    ofile = 'models_' + str(symbol) + '.json' 
    # serialize model to JSON
    model_json = models[symbol].to_json()
    with open(ofile, "w") as json_file:
         json_file.write(model_json)
         # serialize weights to HDF5
         ofile = 'weights_' + str(symbol) + '.h5'
         models[symbol].save_weights(ofile)
    ofile_train = 'x_train_' + str(symbol) + '.json' 
    ofile_test = 'x_test_' + str(symbol) + '.json' 
    pickle.dump(x_trains[symbol],open(ofile_train,'w'))
    pickle.dump(x_tests[symbol],open(ofile_test,'w'))
    ofile_train = 'y_train_' + str(symbol) + '.json' 
    ofile_test = 'y_test_' + str(symbol) + '.json' 
    pickle.dump(y_trains[symbol],open(ofile_train,'w'))
    pickle.dump(y_tests[symbol],open(ofile_test,'w'))     
        
  print("Saved models to disk")  


def predict(keys, models,x_tests):
    y_preds = {}
    for key in keys:
      print key
      if key in models.keys():
        y_preds[key]  = models[key].predict(x_tests[key], verbose=1)
    return y_preds   




def get_MSEs(key, y_tests, y_preds):
    MSE_tests = {} 
    for key in keys:
      if key in y_preds.keys():
       MSE_tests[key] = mean_squared_error(y_tests[key], y_preds[key])
       print MSE_tests[key]
    return MSE_tests     




# transformation of data for plotting
def get_month_symbols(df_lagged, pred_dates, symbols, y_tests, y_preds):
  
  y_hat  = {}
  y = {}
  i = 0
 
  for pred_date in pred_dates:
    
     y_hat[pred_date] = {}
     y[pred_date] = {}
 
     for symbol in symbols:
       print pred_date, symbol
       if symbol in y_preds.keys():
         if (i<len(y_preds[symbol])):     
          y_hat[pred_date][symbol] = y_preds[symbol][i]
         if (i<len(y_tests[symbol])):     
          y[pred_date][symbol] = y_tests[symbol][i]
    
     i = i +1 
  return y, y_hat      


# In[23]:

