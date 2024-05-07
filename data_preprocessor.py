import pandas as pd
from math import ceil
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    def __init__(self):
        pass

    def preprocess_data(self, df):
        """
        Предварительная обработка данных.
        :param df: DataFrame с данными.
        :return: Обработанные данные и целевая переменная.
        """

        # Разделение данных на обучающую и тестовую выборки
        shape = df.shape[0]
        df_new = df[['Close']]
        df_new.head()
        train_set = df_new.iloc[:ceil(shape * 0.75)]
        valid_set = df_new.iloc[ceil(shape * 0.75):]

        return train_set, valid_set

    def preprocess_data_fb_prophet(self, df, shape):
        """
        Предварительная обработка данных для fb_prophet.
        :param df: DataFrame с данными.
        :param shape: df.shape[0].
        :return: Обработанные данные и целевая переменная.
        """

        # Разделение данных на обучающую и тестовую выборки
        df_new = df[['Close']]
        df_new.reset_index(inplace=True)
        df_new['Date'] = pd.to_datetime(df_new.Date, format='%Y-%m-%d')
        df_new.index = df_new['Date']
        df_new.rename(columns={'Close': 'y', 'Date': 'ds'}, inplace=True)
        train_set = df_new.iloc[:ceil(shape * 0.75)]
        valid_set = df_new.iloc[ceil(shape * 0.75):]

        return train_set, valid_set
