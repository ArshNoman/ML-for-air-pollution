from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, svm
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

weather_df = pd.read_csv('Bishkek_PM2.5_2022_YTD.csv')[['Raw Conc.', 'AQI']][:500]
weather_df = weather_df[weather_df["Raw Conc."] >= 0]
weather_df = weather_df[weather_df['AQI'] >= 0]
weather_df.fillna(method='ffill', inplace=True)

# p = plt.plot(weather_df.index, weather_df.values)
# plt.show()

X = np.array(weather_df['Raw Conc.']).reshape(-1, 1)
y = np.array(weather_df['AQI']).reshape(-1, 1)

weather_df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

linearregression = LinearRegression()
linearregression.fit(X_train, y_train)

print("r2 coefficient:", str(r2_score(X_test, y_test)))
print("mean squared error coefficient:", str(mean_squared_error(X_test, y_test)))

y_pred = linearregression.predict(X_test)

plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
plt.show()
