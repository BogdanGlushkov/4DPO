import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import seaborn as sns

rcParams['figure.figsize'] = 20, 10

from data_loader import DataLoader
from models import moving_avg_prediction, linear_regression_prediction, k_nearest_neighbours_predict, \
    auto_arima_prediction, fb_prophet_prediction, lstm_prediction


def main():
    # Загрузка данных
    print("Загрузка данных...")
    data_loader = DataLoader()
    data, target, df = data_loader.load_data()
    print("Данные успешно загружены.")

    plt.plot(data['Close'])
    plt.plot()
    plt.xlabel('Date', size=20)
    plt.ylabel('Stock price', size=20)
    plt.title('Stock Price Prediction by moving Averages', size=20)
    plt.show()

    avg_rms = []

    avg_rms.append(moving_avg_prediction(df))
    avg_rms.append(linear_regression_prediction(df))
    avg_rms.append(k_nearest_neighbours_predict(df))
    avg_rms.append(auto_arima_prediction(df))
    avg_rms.append(fb_prophet_prediction(df))
    avg_rms.append(lstm_prediction(df))

    models = ['Скользящие средние', 'Линейная регрессия', 'К-ближайщих соседей', 'ARIMA', 'Prophet', 'LSTM']

    # Create a bar plot
    plt.bar(models, avg_rms)

    # Add title and labels
    plt.title('RMSE values for different models')
    plt.xlabel('Models')
    plt.ylabel('RMSE, меньше - лучше')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
