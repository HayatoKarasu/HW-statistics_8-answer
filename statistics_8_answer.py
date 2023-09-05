from matplotlib import pyplot as plt

plt.figure(figsize=(10, 10))

plt.text(0.01, 0.9, "Линейная регрессия", fontsize=15, weight='bold')
plt.text(0.35, 0.9, "- это модель машинного обучения, которое", fontsize=15)
plt.text(0.01, 0.8, "предсказывает значение, которое является суммой взвешаных", fontsize=15)
plt.text(0.01, 0.7, "признаков (факторов).", fontsize=15)
form = r"$f(x,b) = x_0 + b_1x_1 + b_2x_2 + ... + b_kx_k$"
plt.text(0.01, 0.6, form, fontsize=15)
plt.text(0.01, 0.5, "Функция потерь", fontsize=15, weight='bold')
plt.text(0.28, 0.5, "- это ошибка прогноза на определенном наборе", fontsize=15)
plt.text(0.01, 0.4, "наблюдаемых данных.", fontsize=15)
form = r"$MSE = \frac{1}{n} \sum_{i=1}^n (Y_i - \hat Y_i)^2$"
plt.text(0.01, 0.3, form, fontsize=15)
plt.text(0.01, 0.2, "Градиентный спуск", fontsize=15, weight='bold')
plt.text(0.32, 0.2, "- это метод нахождения минимума или", fontsize=15)
plt.text(0.01, 0.1, " максимума функции.", fontsize=15)

fig = plt.gca()
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()