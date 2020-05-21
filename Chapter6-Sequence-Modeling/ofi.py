import os
import numpy as np
import pandas as pd

def clean_quotes(quotes):
    quotes = quotes[quotes['Market Flag']=='E']
    quotes = quotes[quotes['Quote Condition'].isnull()]

    quotes['Date'] = pd.to_datetime(quotes['Date'])
    quotes['Time'] = pd.to_timedelta(quotes['Time'])

    return quotes

def save_sp500quotes():
    quotes = pd.read_csv('ES_Quotes.csv')
    dates = quotes['Date'].drop_duplicates()

    half = pd.to_timedelta('14:00:00')
    first_half = clean_quotes(quotes[quotes['Date']==dates[0]])
    for d, next_d in zip(dates[:-1], dates[1:]):
         print(d)
         both_halves = clean_quotes(quotes[quotes['Date']==next_d])
         last_half_bool = both_halves['Time']<=half

         last_half = both_halves[last_half_bool]
         pd.concat([first_half, last_half]).to_pickle(d.replace('/', '') + '_quotes.pickle')

         first_half = both_halves[~last_half_bool]
    # TODO: Consider weekends

def ofi(bid_size, ask_size, bid_price, ask_price):
    ofi_ = pd.Series(0, bid_size.index)

    previous_bid_price = bid_price.shift()
    bid_geq = bid_price >= previous_bid_price
    bid_leq = bid_price <= previous_bid_price

    previous_bid_size = bid_size.shift()
    ofi_[bid_geq] += bid_size[bid_geq]
    ofi_[bid_leq] -= previous_bid_size[bid_leq]

    previous_ask_price = ask_price.shift()
    ask_geq = ask_price >= previous_ask_price
    ask_leq = ask_price <= previous_ask_price

    previous_ask_size = ask_size.shift()
    ofi_[ask_leq] -= ask_size[ask_leq]
    ofi_[ask_geq] += previous_ask_size[ask_geq]

    ofi_.iloc[0] = np.nan
    ofi_.dropna(inplace=True)
    ofi_.name = 'OFI'
    return ofi_

def mid_price(bid_price, ask_price):
    return (bid_price + ask_price)/2

def smart_price(bid_price, bid_size, ask_price, ask_size):
    VWAP = pd.Series((bid_price*bid_size + ask_price*ask_size)/(ask_size+bid_size), bid_size.index, name='VWAP')
    return VWAP

def mid_price_change(bid_price, ask_price):
    dmid = mid_price(bid_price, ask_price).diff()
    dmid.dropna(inplace=True)
    dmid.name = 'Mid price change'
    return dmid

def time_as_index(data, inplace=True):
    if inplace is False:
        raise NotImplementedError
    data.drop_duplicates('Time', keep='last', inplace=True)
    data['Time'] += data['Date']
    data.drop('Date', axis=1, inplace=True)
    data.set_index('Time', inplace=True)

def to_resolution(t, dt):
    return np.ceil(t/dt)*dt

def to_reg_grid(series, dt, agg_fun):
    t0 = np.datetime64(0, 'D')
    time_grid = pd.Series(to_resolution(series.index.values - t0, dt) + t0,
                          series.index.values, name='Time')
    new_series = pd.concat([series, time_grid], axis=1).groupby('Time').agg(agg_fun).iloc[:, 0]
    new_series.name = series.name
    return new_series

if __name__ == '__main__':
    sp500quotes = pd.read_pickle('09012013_quotes.pickle')
    time_as_index(sp500quotes)

    sp500ofi = ofi(sp500quotes['Bid Size'], sp500quotes['Ask Size'],
                   sp500quotes['Bid Price'], sp500quotes['Ask Price'])
    sp500dmid = mid_price_change(sp500quotes['Bid Price'], sp500quotes['Ask Price'])

    print('80th percentile of time deltas:',
          np.percentile(np.diff(sp500quotes.index.values), 80))

    dt = pd.to_timedelta('1s')
    sp500ofi1s = to_reg_grid(sp500ofi, dt, np.sum)
    sp500dmid1s = to_reg_grid(sp500dmid, dt, np.sum)

