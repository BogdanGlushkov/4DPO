import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


class DataLoader:
    def __init__(self):
        pass

    def load_data(self, data_path=None):
        """
        Загрузка данных из файла или генерация случайных данных.
        :param data_path: Путь к файлу с данными. Если None, то генерируются случайные данные.
        :return: DataFrame с данными и целевой переменной.
        """
        if data_path:
            # Загрузка данных из файла
            df = pd.read_csv(data_path)

            df = df.reset_index()

            df['Date'] = df['Date'].astype('datetime64[ns]')

            df.sort_values('Date', ascending=True, inplace=True)
            df.set_index('Date', inplace=True)

            df = df.iloc[:, 1:]

            print(df)

            sns.displot(x='Changes', kde=True, data=df)
            plt.show()

            sns.displot(x='Close', kde=True, data=df)
            plt.show()

            print(df.Changes.describe())

            X = df[['Close']]
            y = df['Changes']

        else:
            # Задаем символ акции (тикер) и диапазон дат
            ticker = 'AAPL'
            start_date = '2015-01-01'
            end_date = '2021-12-31'

            # Получаем исторические данные с помощью pandas_datareader
            df = yf.download(ticker, start=start_date, end=end_date)

            # Отображаем первые несколько строк данных
            df = df.reset_index()
            df['Open'] = df['Open'].astype(float)
            df['High'] = df['High'].astype(float)
            df['Low'] = df['Low'].astype(float)
            df['Close'] = df['Close'].astype(float)
            df['Volume'] = df['Volume'].astype(float)
            df.sort_values('Date', ascending=True, inplace=True)
            df.set_index('Date', inplace=True)

            df['Changes'] = (df['Close'] / df['Close'].shift(1) - 1) * 100
            df['Changes'] = df['Changes'].fillna(0)
            print(df)

            df.to_csv('Data.csv', index=True)

            sns.displot(x='Changes', kde=True, data=df)
            plt.show()

            sns.displot(x='Close', kde=True, data=df)
            plt.show()

            print(df.Changes.describe())

            X = df[['Close']]
            y = df['Changes']

        return X, y, df
