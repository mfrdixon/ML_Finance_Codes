
import csv
import pandas as pd
import numpy as np
import re
import math
import timeit

import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#from utilities import *

# plot the MSEs
def plot_MSEs(pred_dates, y, y_hat):

    str_dates = []

    for pred_date in pred_dates:
        dt= pd.Timestamp(np.datetime64(pred_date)).to_pydatetime()
        str_dates.append(str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day))
    print str_dates
    MSE_tests_date = {}
    y_mean = {}
    y_hat_mean = {}
    for pred_date in pred_dates:
      #MSE_trains[symbol]= mean_squared_error(y_trains[symbol], pred_trains[symbol])  
      if pred_date in y_hat.keys():
       y_mean[pred_date] = np.mean(np.concatenate(y[pred_date].values()))
       y_hat_mean[pred_date] = np.mean(np.concatenate(y_hat[pred_date].values()))
       MSE_tests_date[pred_date] = mean_squared_error(np.concatenate(y[pred_date].values()), np.concatenate(y_hat[pred_date].values()))
       print MSE_tests_date[pred_date]

    plt.plot(range(0,len(pred_dates)), y_mean.values(), label="mean excess returns")
    plt.plot(range(0,len(pred_dates)), y_hat_mean.values(), label="mean predicted excess returns")
    plt.plot(range(0,len(pred_dates)), MSE_tests_date.values(), label="MSE (" + str(np.round(np.mean(MSE_tests_date.values()),3))+")")

    plt.xlabel('Period', fontsize=20)
    plt.xticks(range(0,len(pred_dates)), str_dates,  rotation='vertical')
    plt.ylabel('MSE', fontsize=20)
    plt.title('MSE in Excess Returns (RNN)')
    plt.legend()


# In[22]:

# plot the excess returns
def plot_excess_returns(pred_dates, y, y_hat):

    results  = {}
    n_ins = {}
    fig = plt.figure(figsize=(16,9))  
    str_dates = []

    for pred_date in pred_dates:
        dt= pd.Timestamp(np.datetime64(pred_date)).to_pydatetime()
        str_dates.append(str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day))

    

    for q in [10,15,20,25]:
      results[q] = []
      n_ins[q] = []
 
      for pred_date in pred_dates:
        if pred_date in y_hat.keys():    
          #print pred_date
          y_hat_ = np.concatenate(y_hat[pred_date].values())  
          n = len(y_hat_)
          m = np.int(n*q/100.0)
          #print n, m 
          idx = np.argsort(y_hat_)[::-1][:m]
          #y_hat_top = np.array(y_hat[pred_date].values())[idx]
          y_top = np.concatenate(y[pred_date].values())[idx]
          results[q].append(np.mean(y_top))
          n_ins[q].append(m)  
          #print pred_date, np.mean(y_top), n, q
        else:
          print "missing date: " + pred_date  
  
      str_q = str(q) + "%: " + str(round(4.0*np.mean(results[q]),3)) + " (" + str(int(np.mean(n_ins[q]))) + ")"  
      plt.plot(range(0,len(pred_dates)), results[q], label=str_q)
    
    plt.xlabel('Period', fontsize=20)
    plt.xticks(range(0,len(pred_dates)), str_dates,  rotation='vertical')
    plt.ylabel('Excess returns', fontsize=20)
    plt.title('Annualized portfolio excess returns')
    plt.legend()
    return results


# In[21]:

# plot the information ratios
def plot_IRs(pred_dates, y, y_hat):
    
    n_ins = {}
    IRs = {}  # information ratios
    fig = plt.figure(figsize=(16,9))  
    str_dates = []
    eps = 1e-3

    for pred_date in pred_dates:
        dt= pd.Timestamp(np.datetime64(pred_date)).to_pydatetime()
        str_dates.append(str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day))

    for q in [10,15,20,25]:
  
      n_ins[q] = []
      IRs[q] = []  
 
      for pred_date in pred_dates:
        if pred_date in y_hat.keys():
          #print pred_date
          y_hat_ = np.concatenate(y_hat[pred_date].values())  
          n = len(y_hat_)
          m = np.int(n*q/100.0)
          #print n, m 
          #print np.mean(y_hat[pred_date].values()), np.std(y_hat[pred_date].values())
          #print np.mean(y[pred_date].values()), np.std(y[pred_date].values())
          idx = np.argsort(y_hat_)[::-1][:m]
        
          y_top = np.concatenate(y[pred_date].values())[idx]
        
          #print np.mean(y_top), np.std(y_top)
          IRs[q].append(2.0*np.mean(y_top)/(np.std(y_top) + eps)) # annualized
          n_ins[q].append(m)  
          #print pred_date, np.mean(y_top), n, q
  
      str_q = str(q) + "%: " + str(round(4.0*np.mean(IRs[q]),3)) + " (" + str(int(np.mean(n_ins[q]))) + ")"  
      plt.plot(range(0,len(pred_dates)), IRs[q], label=str_q)
    
    plt.xlabel('Period', fontsize=20)
    plt.xticks(range(0,len(pred_dates)), str_dates,  rotation='vertical')
    plt.ylabel('Information Ratio', fontsize=20)
    plt.title('Information Ratios')
    plt.legend()
    return IRs