import sqlite3
import pandas as pd
import os

from functools import reduce

'''
This scripts does all the preparatory work needed to be done for every single
DataTable from the DB.
'''

# Change working directory to this script's location, for the purposes of importing
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_TrainingDataFrame():
    # General connection to the DB with data
    StockDataDatabase = sqlite3.connect("StockData.db")

    # Index Data (NASDAQ and S&P 500)
    NASDAQ = pd.read_sql("SELECT * FROM Indexes WHERE IndexName='NASDAQ' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    SP500 = pd.read_sql("SELECT * FROM Indexes WHERE IndexName='S&P 500' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])

    # Stocks' Tickers (65 US Tech Stocks)
    tickers = pd.read_sql("SELECT DISTINCT * FROM Tickers", con=StockDataDatabase)

    # Market Indicators
    VIX = pd.read_sql("SELECT * FROM MarketIndicators WHERE Indicator='VIX' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    bond_13_Week = pd.read_sql("SELECT * FROM MarketIndicators WHERE Indicator='13-Week Treasury' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    bond_5_Year = pd.read_sql("SELECT * FROM MarketIndicators WHERE Indicator='5-Year Treasury' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])

    # Currency exchange rates
    EUR_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='EUR/USD' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    JPY_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='JPY/USD' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    GBP_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='GBP/USD' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    USD_CHF = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='USD/CHF' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    AUD_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='AUD/USD' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    USD_CAD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='USD/CAD' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])
    NZD_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='NZD/USD' AND Date<='2025-09-04'", con=StockDataDatabase, parse_dates=["Date"])

    '''
    Based on analysis in TradingDays.ipynb, there is a mismatch between trading days
    on Forex and Stock/Macroeconomic Market. Based on analysis, we have to clean up the mess :)
    '''

    #### MISSING VALUES ####

    # Dates which do not have datarow in Forex data
    missing_dates = [pd.Timestamp('2017-11-16 00:00:00'), pd.Timestamp('2019-05-22 00:00:00'), pd.Timestamp('2025-04-21 00:00:00')]
    # First available data for missing_dates in history
    previous_days = [pd.Timestamp('2017-11-15 00:00:00'), pd.Timestamp('2019-05-21 00:00:00'), pd.Timestamp('2025-04-17 00:00:00')]

    # Filling missing data with previous_days data
    for index in range(len(previous_days)):
        for forex in [EUR_USD, JPY_USD, GBP_USD, USD_CHF, AUD_USD, USD_CAD, NZD_USD]:
            new_row = forex[forex["Date"] == previous_days[index]].copy()
            new_row["Date"] = missing_dates[index]

            forex = pd.concat([forex, new_row], ignore_index=True)
    
    forex_dict = {
        "EUR_USD": EUR_USD,
        "JPY_USD": JPY_USD,
        "GBP_USD": GBP_USD,
        "USD_CHF": USD_CHF,
        "AUD_USD": AUD_USD,
        "USD_CAD": USD_CAD,
        "NZD_USD": NZD_USD
    }

    for name, dataframe in forex_dict.items():
        original_df = dataframe.copy()  # Keep a copy for selection
    
        for index in range(len(previous_days)):
            # Select only from the original DataFrame
            new_row = original_df[original_df["Date"] == previous_days[index]].copy()
            if new_row.empty:
                continue
        
            new_row["Date"] = missing_dates[index]
            original_df = pd.concat([original_df, new_row], ignore_index=True)

        # Sort by Date and reset index
        original_df = original_df.sort_values(by="Date").reset_index(drop=True)

        forex_dict[name] = original_df

    # Assign back to original variables
    EUR_USD = forex_dict["EUR_USD"]
    JPY_USD = forex_dict["JPY_USD"]
    GBP_USD = forex_dict["GBP_USD"]
    USD_CHF = forex_dict["USD_CHF"]
    AUD_USD = forex_dict["AUD_USD"]
    USD_CAD = forex_dict["USD_CAD"]
    NZD_USD = forex_dict["NZD_USD"]

    #### SHIFTS ####

    # Now we perform shifts in the data, so we obtain lagged features for the prediction
    initial_columns = ["Open", "High", "Low", "Close", "Volume"]
    data_types = ["VIX", "Bond13W", "Bond5Y", "NASDAQ", "SP500", "EUR/USD", "JPY/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
    historic_columns = []

    for shift in range(1, 10):
        for column in initial_columns:
            # Market Indicators
            VIX[f"{column}_{shift}"] = VIX[f"{column}"].shift(shift)
            bond_13_Week[f"{column}_{shift}"] = bond_13_Week[f"{column}"].shift(shift)
            bond_5_Year[f"{column}_{shift}"] = bond_5_Year[f"{column}"].shift(shift)

            # Indices
            NASDAQ[f"{column}_{shift}"] = NASDAQ[f"{column}"].shift(shift)
            SP500[f"{column}_{shift}"] = SP500[f"{column}"].shift(shift)

            # Currency exchanges
            EUR_USD[f"{column}_{shift}"] = EUR_USD[f"{column}"].shift(shift)
            JPY_USD[f"{column}_{shift}"] = JPY_USD[f"{column}"].shift(shift)
            GBP_USD[f"{column}_{shift}"] = GBP_USD[f"{column}"].shift(shift)
            USD_CHF[f"{column}_{shift}"] = USD_CHF[f"{column}"].shift(shift)
            AUD_USD[f"{column}_{shift}"] = AUD_USD[f"{column}"].shift(shift)
            USD_CAD[f"{column}_{shift}"] = USD_CAD[f"{column}"].shift(shift)
            NZD_USD[f"{column}_{shift}"] = NZD_USD[f"{column}"].shift(shift)
            
            for data_type in data_types:
                historic_columns.append(f"{column}_{shift}_{data_type}")

    VIX = VIX.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_VIX" if x != "Date" else x)
    bond_13_Week = bond_13_Week.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_Bond13W" if x != "Date" else x)
    bond_5_Year = bond_5_Year.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_Bond5Y" if x != "Date" else x)

    NASDAQ = NASDAQ.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_NASDAQ" if x != "Date" else x)
    SP500 = SP500.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_SP500" if x != "Date" else x)

    EUR_USD = EUR_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_EUR/USD" if x != "Date" else x)
    JPY_USD = JPY_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_JPY/USD" if x != "Date" else x)
    GBP_USD = GBP_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_GBP/USD" if x != "Date" else x)
    USD_CHF = USD_CHF.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_USD/CHF" if x != "Date" else x)
    AUD_USD = AUD_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_AUD/USD" if x != "Date" else x)
    USD_CAD = USD_CAD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_USD/CAD" if x != "Date" else x)
    NZD_USD = NZD_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_NZD/USD" if x != "Date" else x)

    #### FINAL CONCAT ####

    TradingDataFrames = [VIX, bond_13_Week, bond_5_Year, NASDAQ,
                        SP500, EUR_USD, JPY_USD, GBP_USD, USD_CHF, 
                        AUD_USD, USD_CAD, NZD_USD]

    TrainingDataFrame = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), TradingDataFrames)[["Date"] + historic_columns]

    return TrainingDataFrame, tickers, historic_columns, StockDataDatabase

def generate_TestDataFrame():
    # General connection to the DB with data
    StockDataDatabase = sqlite3.connect("StockData.db")

    # Index Data (NASDAQ and S&P 500)
    NASDAQ = pd.read_sql("SELECT * FROM Indexes WHERE IndexName='NASDAQ' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    SP500 = pd.read_sql("SELECT * FROM Indexes WHERE IndexName='S&P 500' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])

    # Stocks' Tickers (65 US Tech Stocks)
    tickers = pd.read_sql("SELECT * FROM Tickers", con=StockDataDatabase)

    # Market Indicators
    VIX = pd.read_sql("SELECT * FROM MarketIndicators WHERE Indicator='VIX' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    bond_13_Week = pd.read_sql("SELECT * FROM MarketIndicators WHERE Indicator='13-Week Treasury' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    bond_5_Year = pd.read_sql("SELECT * FROM MarketIndicators WHERE Indicator='5-Year Treasury' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])

    # Currency exchange rates
    EUR_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='EUR/USD' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    JPY_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='JPY/USD' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    GBP_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='GBP/USD' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    USD_CHF = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='USD/CHF' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    AUD_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='AUD/USD' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    USD_CAD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='USD/CAD' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])
    NZD_USD = pd.read_sql("SELECT * FROM CurrencyExchange WHERE Currencies='NZD/USD' AND Date>='2022-11-01'", con=StockDataDatabase, parse_dates=["Date"])

    #### SHIFTS ####

    # Now we perform shifts in the data, so we obtain lagged features for the prediction
    initial_columns = ["Open", "High", "Low", "Close", "Volume"]
    data_types = ["VIX", "Bond13W", "Bond5Y", "NASDAQ", "SP500", "EUR/USD", "JPY/USD", "GBP/USD", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"]
    historic_columns = []

    for shift in range(1, 10):
        for column in initial_columns:
            # Market Indicators
            VIX[f"{column}_{shift}"] = VIX[f"{column}"].shift(shift)
            bond_13_Week[f"{column}_{shift}"] = bond_13_Week[f"{column}"].shift(shift)
            bond_5_Year[f"{column}_{shift}"] = bond_5_Year[f"{column}"].shift(shift)

            # Indices
            NASDAQ[f"{column}_{shift}"] = NASDAQ[f"{column}"].shift(shift)
            SP500[f"{column}_{shift}"] = SP500[f"{column}"].shift(shift)

            # Currency exchanges
            EUR_USD[f"{column}_{shift}"] = EUR_USD[f"{column}"].shift(shift)
            JPY_USD[f"{column}_{shift}"] = JPY_USD[f"{column}"].shift(shift)
            GBP_USD[f"{column}_{shift}"] = GBP_USD[f"{column}"].shift(shift)
            USD_CHF[f"{column}_{shift}"] = USD_CHF[f"{column}"].shift(shift)
            AUD_USD[f"{column}_{shift}"] = AUD_USD[f"{column}"].shift(shift)
            USD_CAD[f"{column}_{shift}"] = USD_CAD[f"{column}"].shift(shift)
            NZD_USD[f"{column}_{shift}"] = NZD_USD[f"{column}"].shift(shift)
            
            for data_type in data_types:
                historic_columns.append(f"{column}_{shift}_{data_type}")

    VIX = VIX.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_VIX" if x != "Date" else x)
    bond_13_Week = bond_13_Week.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_Bond13W" if x != "Date" else x)
    bond_5_Year = bond_5_Year.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_Bond5Y" if x != "Date" else x)

    NASDAQ = NASDAQ.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_NASDAQ" if x != "Date" else x)
    SP500 = SP500.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_SP500" if x != "Date" else x)

    EUR_USD = EUR_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_EUR/USD" if x != "Date" else x)
    JPY_USD = JPY_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_JPY/USD" if x != "Date" else x)
    GBP_USD = GBP_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_GBP/USD" if x != "Date" else x)
    USD_CHF = USD_CHF.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_USD/CHF" if x != "Date" else x)
    AUD_USD = AUD_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_AUD/USD" if x != "Date" else x)
    USD_CAD = USD_CAD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_USD/CAD" if x != "Date" else x)
    NZD_USD = NZD_USD.dropna().reset_index(drop=True).rename(columns=lambda x: x + "_NZD/USD" if x != "Date" else x)

    #### FINAL CONCAT ####

    TradingDataFrames = [VIX, bond_13_Week, bond_5_Year, NASDAQ,
                        SP500, EUR_USD, JPY_USD, GBP_USD, USD_CHF, 
                        AUD_USD, USD_CAD, NZD_USD]

    TrainingDataFrame = reduce(lambda left, right: pd.merge(left, right, on="Date", how="inner"), TradingDataFrames)[["Date"] + historic_columns]

    return TrainingDataFrame, tickers, historic_columns, StockDataDatabase