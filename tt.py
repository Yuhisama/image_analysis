import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 生成樣本數據
np.random.seed(0)
data = np.random.normal(loc=0, scale=1, size=1000)

# 計算概率密度函數
mu, std = norm.fit(data)
xmin, xmax = min(data), max(data)
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)

# 繪製概率密度函數圖表
plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, density=True, alpha=0.6, color='g')  # 繪製直方圖
plt.plot(x, p, 'k', linewidth=2)  # 繪製概率密度函數
plt.title("Probability Density Function")
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.show()
