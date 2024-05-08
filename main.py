import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

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
    plt.title('Stock Price AAPL', size=20)
    plt.show()

    avg_rms = []

    avg_rms.append([moving_avg_prediction(df), 'Скользящие средние'])
    avg_rms.append([linear_regression_prediction(df), 'Линейная регрессия'])
    avg_rms.append([k_nearest_neighbours_predict(df), 'К-ближайщих соседей'])
    avg_rms.append([fb_prophet_prediction(df), 'Prophet'])
    avg_rms.append([auto_arima_prediction(df), 'ARIMA'])
    avg_rms.append([lstm_prediction(df), 'LSTM'])

    sorted_avg_rms = sorted(avg_rms, key=lambda x: x[0])

    models = []
    avg_scores = []

    for i in range(len(sorted_avg_rms)):
        avg_scores.append(sorted_avg_rms[i][0])
        models.append(sorted_avg_rms[i][1])

    # Create a bar plot
    plt.bar(models, avg_scores)

    # Add title and labels
    plt.title('RMSE values for different models')
    plt.xlabel('Models')
    plt.ylabel('RMSE, меньше - лучше')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
