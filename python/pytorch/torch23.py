
from pandas import read_csv
from datetime import datetime
from pandas import DataFrame
from statsmodels.tsa.arima.model import ARIMA
from matplotlib import pyplot


def parser(x):
    return datetime.strptime('199'+x, '%Y-%m')

series = read_csv('C:\chatbot\python\pythorch\data\sales.csv', header=0, parse_dates=[0], index_col=0,
                  squeeze=True, date_parser=parser)
model = ARIMA(series, order=(5,1,0))
model_fit = model.fit()
print(model_fit.summary())
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())