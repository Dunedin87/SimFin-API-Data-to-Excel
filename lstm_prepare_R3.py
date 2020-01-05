import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import datetime as dt
import tiingo
import warnings
warnings.filterwarnings("ignore")

'''
create_df creates a pandas database of a particular ticker
the module runner takes all the ticker from SimFin's bulk data and creates a csv
using pandas dataframe created from create_df. There is a attribute list given to runner
such as revenue, net income etc. If these attributes are not found in a particular ticker's
dataset, it is not included.

All the csvs are added to a path under the name Ticker + _lstm.csv

the module csv_reader then reads all the csv in a particular folder. It returns a batch
pytorch tensor with all the data
the shape of tensor is [num_tickers, attributes, financials_periods]

lstm will need sequence length (financial periods) to be the same. Data for avaiable financial
periods are read, then the rest is padded with 0s till max length. Let's say 8 quarters of data
is avaialble, and max_cols is 50, the rest of 42 columns will be padded with zero

Also previous runner module should have ensured that all the csv had same attribute in the
same sequence. example, revenue, net income, assets etc.

now we need to create something that creates target variable.
create_prices module creates a csv with index of financial period and tickers as columns.
can use this to create target variable Y

'''
incomestatement = pd.read_csv('data//us-income-quarterly.csv',sep=';')
balancesheet = pd.read_csv('data//us-balance-quarterly.csv',sep=';')
cashflow = pd.read_csv('data//us-cashflow-quarterly.csv',sep=';')


def create_df(ticker):

    df = incomestatement.loc[incomestatement['Ticker']== ticker]
    df['Period'] = df[['Fiscal Year','Fiscal Period']].apply(lambda x: ''.join(x.map(str)),axis=1)
    df.set_index('Period',inplace=True)
    df2 = balancesheet.loc[balancesheet['Ticker']== ticker]
    df2['Period'] = df2[['Fiscal Year','Fiscal Period']].apply(lambda x: ''.join(x.map(str)),axis=1)
    df2.set_index('Period',inplace=True)
    df3 = cashflow.loc[balancesheet['Ticker']== ticker]
    df3['Period'] = df3[['Fiscal Year','Fiscal Period']].apply(lambda x: ''.join(x.map(str)),axis=1)
    df3.set_index('Period',inplace=True)

    df_all = pd.merge(df,df2,on='Period')
    df_all = pd.merge(df_all,df3, on='Period')
    df_all.drop(columns=['Fiscal Year','Fiscal Period','Ticker_x','SimFinId_x','Currency_x','Fiscal Year_x','Fiscal Period_x','Report Date_x','Publish Date_x','Shares (Basic)_x','Shares (Diluted)_x','Depreciation & Amortization_x','Ticker_y', 'SimFinId_y', 'Currency_y',
       'Fiscal Year_y', 'Fiscal Period_y', 'Report Date_y',
       'Publish Date_y', 'Shares (Basic)_y', 'Shares (Diluted)_y','Depreciation & Amortization_y'], inplace=True)
    df_all2 = df_all.transpose()
    df_all2 = df_all2.reindex(sorted(df_all2.columns), axis=1)

    return df_all2


def csv_maker(path='data//lstm//',attr_list = None):
    all_tickers = incomestatement['Ticker'].unique()
    not_added = []

    if attr_list == None:
        attr_list = ['Revenue','Cost of Revenue','Gross Profit','Operating Expenses','Operating Income (Loss)',
                          'Interest Expense, Net','Net Income',
                           'Cash, Cash Equivalents & Short Term Investments','Total Current Assets',
                            'Property, Plant & Equipment, Net','Total Noncurrent Assets','Total Assets',
                             'Total Current Liabilities','Total Liabilities','Retained Earnings',
                            'Total Equity','Net Cash from Operating Activities','Net Cash from Investing Activities',
                         'Net Change in Cash','Shares (Basic)']

    for ticker in all_tickers:
        df = create_df(ticker)
        df = df.drop(labels=['Ticker','Currency','Report Date', 'Publish Date']).astype(float)
        df = df.loc[attr_list]
        df.dropna(inplace=True)

        if attr_list == df.index.tolist():
            df2 = df.loc[['Revenue', 'Shares (Basic)']].copy()
            df.drop(['Shares (Basic)'], inplace=True)
            df = df/df.loc['Revenue'][0]
            df.loc['Revenue Actual'] = df2.loc['Revenue']
            df.loc['Shares (Basic)'] = df2.loc['Shares (Basic)']
            df.dropna(axis = 1, inplace=True)
            if len(df.columns) > 0:
                df.to_csv(path+'{}_lstm.csv'.format(ticker))
        else:
            not_added.append(ticker)
    # print('Following list of tickers not added as some attribute values were missing', not_added)
    return not_added


def csv_to_tensor(file_list=None, path='data//lstm//',price_data='data//updated_prices.csv'):
    max_columns = 55 # Number of maximum columns to be padded to
    all_tensor = torch.Tensor()
    all_mcap = torch.Tensor()
    target = torch.Tensor()

    if file_list is None:
        file_list = os.listdir(path)

    prices = pd.read_csv(price_data,index_col=0).transpose()
    prices.fillna(0,inplace=True)

    for file in file_list:
        ticker = file.split('_')[0]
        df = pd.read_csv(path + file,index_col=0)
        revenue = df.loc['Revenue Actual'].copy()
        shares = df.loc['Shares (Basic)'].copy()
        df.drop(['Revenue Actual','Shares (Basic)'], inplace = True)
        mcap = prices.loc[ticker][df.columns.values] * shares
        mcap /= mcap[0]
        mcap_tensor = torch.Tensor(mcap.tolist()).view(1,-1)
        mcap_pad = torch.zeros(1,max_columns-mcap_tensor.size()[1])
        mcap_tensor = torch.cat((mcap_tensor,mcap_pad),dim=1)
        all_mcap = torch.cat((all_mcap,mcap_tensor),dim=0)

        # Padd extra columns with value of 0, since companies will have varying number of financial periods of data
        # Setting max columns as 50
        num_columns = len(df.columns)
        pad = torch.zeros(1,len(df),max_columns-num_columns)
        tensor = torch.Tensor(df.to_numpy().reshape(1,df.to_numpy().shape[0],-1))
        tensor = torch.cat((tensor,pad),dim=2)
        all_tensor = torch.cat((all_tensor,tensor),dim=0)

        #Convert any nans value in mcap to 0

    all_mcap[all_mcap != all_mcap] = 0
    all_mcap[all_mcap == float('Inf')] = 0
    all_mcap[all_mcap == float('-Inf')] = 0
    all_mcap[all_mcap < 0] = 0

    all_tensor[all_tensor != all_tensor] = 0
    all_tensor[all_tensor == float('Inf')] = 0
    all_tensor[all_tensor == float('-Inf')] = 0

    all_tensor = all_tensor.permute(2,0,1)
    return all_tensor, all_mcap


list_attribute = ['Revenue','Cost of Revenue','Selling, General & Administrative','Research & Development',
                  'Gross Profit','Operating Expenses','Operating Income (Loss)',
                  'Interest Expense, Net','Net Income',
                   'Cash, Cash Equivalents & Short Term Investments','Total Current Assets',
                    'Property, Plant & Equipment, Net','Total Noncurrent Assets','Total Assets',
                     'Total Current Liabilities','Total Liabilities','Retained Earnings',
                    'Total Equity','Net Cash from Operating Activities','Net Cash from Investing Activities',
                 'Net Change in Cash']

# runner(path='data//lstm_test2//', attr_list=list_attribute)

# tensors = csv_reader(files, path='data//lstm_test/')

# print(tensors.size())

def get_tiingo_data(tickers, start_year, end_year, save_file = None):

    tiingo_api = '510e991ef5b36d4a159292fde1d3e9f8af5d5e9e'
    config = {}
    config['api_key'] = tiingo_api
    config['session'] = True
    client = tiingo.TiingoClient(config)

    #Returns dataframe with
    start_date = dt.date(start_year,1,1)
    current_date = dt.date.today()
    all_prices = client.get_dataframe(tickers,frequency='daily',metric_name='adjClose',startDate=start_date,endDate=current_date)

    if save_file:
        all_prices.to_csv(save_file)

    return all_prices

def create_prices(file_path, prices_local = 'bulk', tickers=None, start_year = 2005, end_year = 2019):

    start_date = dt.date(start_year,1,1)
    current_date = dt.date.today()

    files = os.listdir(file_path)
    if tickers == None:
        tickers = []
        for f in files:
            tickers.append(f.split('_')[0])

    all_dates = []

    if prices_local == 'bulk':
        # all_prices = get_tiingo_data(tickers,start_year,end_year, save_file = 'data//all_pric.csv')
        all_prices = simfin_price('data//us-shareprices-daily.csv')
    else:
        all_prices = pd.read_csv(prices_local, index_col=0, parse_dates=True)

    #Ensuring all dates are present, otherwise if a date falls on non-market day (weekend, holiday),
    #It will not show

    all_prices = all_prices.reindex(pd.date_range(start_date,current_date,freq='D',tz = all_prices.index.tz), method='bfill')

    df_subset = pd.DataFrame(columns = all_prices.columns)

    for i in range(start_year,end_year+1):
        for j in [1,3,6,9]:
            q_dict = {1:'Q1',3:'Q2',6:'Q3',9:'Q4'}
            quarter = q_dict[j]
            d = dt.date(i,j,1)
            df_subset.loc[str(i)+quarter] = all_prices.loc[d]
    return df_subset

def simfin_price(bulk_data_path):
    current_date = dt.date.today()
    start_date = dt.date(2007,1,1)
    df = pd.read_csv(bulk_data_path, sep=';')
    prices = df[['Ticker','Date','Adj. Close']]
    prices = pd.pivot_table(prices,values='Adj. Close', index = 'Date', columns = ['Ticker'])
    prices.index = pd.to_datetime(prices.index)
    prices = prices.reindex(pd.date_range(start_date,current_date,freq='D'), method='bfill')
    return prices


x, y = csv_to_tensor()

torch.save(x, 'data//x_tensor_R8.pt')
torch.save(y, 'data//y_tensor_R8.pt')

print(x.size())
print(y.size())
