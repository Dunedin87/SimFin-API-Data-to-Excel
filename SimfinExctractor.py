import json, os, requests
import pandas as pd
import numpy as np
from urllib.request import urlopen

APIKEY = 'Get'
Quarters = ['Q1', 'Q2', 'Q3', 'Q4']


def datatoexcel(data, filename, verbose=False):
    data_export = data.copy()

    if os.path.isfile(filename):
        existing_data = pd.read_excel(filename)
        data_columns = data.columns.values
        additional_info = np.array([])
        existing_data_columns = existing_data.columns.values
        existing_data_information = existing_data['Information'].values
        data_information = data['Information'].values
        duplicate_columns = np.intersect1d(existing_data_columns, data_columns)

        if len(data.drop(labels=duplicate_columns, axis=1).columns.values) == 0:
            data.drop(labels='Information', axis=1, inplace=True)
        else:
            data.drop(labels=duplicate_columns, axis=1, inplace=True)

        # Check if the values on the information column is the same
        if np.array_equal(existing_data_information, data_information) == True:
            pd.concat([existing_data, data], axis=1).to_excel(filename)
            if verbose == True:
                print('Existing File found, Updated information')
        else:
            existing_data[data.columns.values[0]] = np.nan

            # This is if the values in the information columns are not the same

            if verbose == True:
                print('Additional Value Module')

            if len(existing_data_information) > len(data_information):
                for i in range(len(existing_data_columns)):
                    if (existing_data_information[i] in data_information) == False:
                        additional_info = np.append(additional_info, existing_data_information[i])
            else:
                for i in range(len(data_information)):
                    if (data_information[i] in existing_data_information) == False:
                        additional_info = np.append(additional_info, data_information[i])

            if verbose == True:
                print('Additional Values in the dataset: ', additional_info)

            # Both arrays should have at least each other values. Additional_vals has the values
            # not in one of them.

            # Following for loop matches existing values and adds to existing_data

            for i in range(len(existing_data)):
                for j in range(len(data)):
                    if (existing_data['Information'].iloc[i]) == (data_export['Information'].iloc[j]):
                        existing_data.loc[i, data.columns.values[0]] = data_export[data.columns.values[0]].iloc[j]

            additional_info_pd = pd.DataFrame(additional_info, columns=['Information'])
            additional_vals_pd = pd.DataFrame(columns=existing_data.columns.values).drop(columns=['Information'])
            add_all = pd.concat([additional_info_pd, additional_vals_pd], axis=1)

            # After this a dataframe 'add_all' is created that can be concated to the original
            # dataframe. only contains values with 'additional value attributes'

            for i in range(len(add_all['Information'])):
                for j in range(len(data)):
                    if (add_all['Information'].iloc[i]) == (data_export['Information'].iloc[j]):
                        add_all.loc[i, data.columns.values] = data.iloc[j]

            pd.concat([existing_data, add_all], axis=0, sort=False, ignore_index=True).to_excel(filename)

    else:
        print('Created new file ', str(filename))
        data_export.to_excel(filename)


def SaveDataToDisk(data, filename):
    with open(filename + '.json', 'w') as f:
        json.dump(data, f, indent=2)


def ExctractStandardData(data_json, ticker, year, ptype):
    quarter = ptype
    translation = {'9M':'Q3','HY':'Q2','FY':'Q4'}

    if quarter == '9M':
        quarter = 'Q3'
    elif quarter == 'HY':
        quarter = 'Q2'
    elif quarter == 'FY':
        quarter = 'Q4'

    data_dict = dict()

    for i in range(len(data_json['values'])):
        data_dict[data_json['values'][i]['standardisedName']] = data_json['values'][i]['valueChosen']

    Info_Period = str(year) + str(quarter)
    pd_data = pd.DataFrame(columns=['Information', Info_Period])
    pd_data['Information'] = data_dict.keys()
    pd_data[Info_Period] = data_dict.values()

    return pd_data


def GetID(ticker, save=False):
    if os.path.isfile('AllTickers.xlsx'):
        all_tickers = pd.read_excel('AllTickers.xlsx')
        all_tickers.set_index('ticker', inplace=True)
        simid = all_tickers.loc[ticker]['simId']
    else:
        query = 'https://simfin.com/api/v1/info/find-id/ticker/' + ticker + '?api-key=' + APIKEY

        with open(query) as f:
            all_t = f.read()
            all_tick = json.loads(all_t)

        simid = all_tick[0]['simId']

    return str(simid)


def GetStatements(ticker, save=True):
    query = 'https://simfin.com/api/v1/companies/id/' + GetID(ticker) + '/statements/list' + '?api-key=' + APIKEY

    print(query)


def GetFinancialData(ticker, ptype='FY', year=2017, save=False, toexcel=False):
    CompanyID = GetID(ticker)
    StockPD = pd.DataFrame()

    query = 'https://simfin.com/api/v1/companies/id/' + CompanyID + '/statements/standardised'
    IS_String = query + '?stype=pl' + '&ptype=' + ptype + '&fyear=' + str(year) + '&api-key=' + APIKEY
    with urlopen(IS_String) as response:
        IS = response.read()
        IncomeStatement = json.loads(IS)

    BS_String = query + '?stype=bs' + '&ptype=' + ptype + '&fyear=' + str(year) + '&api-key=' + APIKEY
    with urlopen(BS_String) as response:
        BS = response.read()
        BalanceSheet = json.loads(BS)

    CF_String = query + '?stype=cf' + '&ptype=' + ptype + '&fyear=' + str(year) + '&api-key=' + APIKEY
    with urlopen(CF_String) as response:
        CF = response.read()
        CashFlow = json.loads(CF)

    IncomeStatementPD = ExctractStandardData(IncomeStatement, ticker, year, ptype)
    CashFlowPD = ExctractStandardData(CashFlow, ticker, year, ptype)
    BalanceSheetPD = ExctractStandardData(BalanceSheet, ticker, year, ptype)

    StockPD = pd.concat([IncomeStatementPD, CashFlowPD, BalanceSheetPD], axis=0, ignore_index=True)

    if save == True:
        SaveDataToDisk(IncomeStatement, ticker + str(year) + 'IncomeState')
        SaveDataToDisk(BalanceSheet, ticker + str(year) + 'BalanceSheet')
        SaveDataToDisk(CashFlow, ticker + str(year) + 'CashFlow')
    if toexcel == True:
        datatoexcel(StockPD, ticker + '.xlsx')

    return StockPD

    # MSFT = GetFinancialData('MSFT',year=2012,ptype='Q4',toexcel=True)
    # stocks = ['IBM', 'XOM','CVX']


def make_ratios(dataframe):

    dataframe.set_index('Information', inplace=True)
    for i in range(len(dataframe)):
        name = str(dataframe.index.values[i] + ' Ratio')
        try:
            dataframe.loc[name] = dataframe.iloc[i] / dataframe.iloc[i].iloc[-1]
        except:
            print('0 Value')

    return dataframe


#
# for i in range(2011, 2019):
#     for j in Quarters:
#         print('Getting info for year {}, quarter {}'.format(i, j))
#         GetFinancialData('WMT', year=i, ptype=j, toexcel=True)

WMT3 = pd.read_excel('WMT.xlsx')
WMT2 = make_ratios(WMT3)
WMT2.to_excel('WMT2.xlsx')
