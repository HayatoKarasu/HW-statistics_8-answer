import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

#Создадим сэмпл
n_samples = 500

age_owner = np.random.choice(90, n_samples) + 21
lenght = np.random.choice(120, n_samples) + 15
width = np.random.choice(80, n_samples) + 10

price = lenght * width * 100 + 126

data = pd.DataFrame({'age_owner': age_owner, 'lenght': lenght, 'width': width, 'price': price})
print(data.head(5))

X = data[['age_owner', 'lenght', 'width']]
y = data['price']
reg = LinearRegression().fit(X, y)
print('Weights: {}'.format(reg.coef_))
print('Bias: {}'.format(reg.intercept_))

pred_values = reg.predict(data[['age_owner', 'lenght', 'width']])
print('Error: {}'.format(mean_absolute_error(pred_values, y)))

#уберем возраст
X = data[['lenght', 'width']]
y = data['price']
reg = LinearRegression().fit(X, y)
print('Weights: {}'.format(reg.coef_))
print('Bias: {}'.format(reg.intercept_))

pred_values = reg.predict(data[['lenght', 'width']])
print('Error: {}'.format(mean_absolute_error(pred_values, y)))
print(data.price.mean())

#создадим новый признак
data['mult'] = data['lenght'] * data['width']
print(data.head(5))

X = data[['mult']]
y = data['price']
reg = LinearRegression().fit(X, y)
print('Weights: {}'.format(reg.coef_))
print('Bias: {}'.format(reg.intercept_))

pred_values = reg.predict(data[['mult']])
print('Error: {}'.format(mean_absolute_error(pred_values, y)))

print(data[['mult', 'price']])