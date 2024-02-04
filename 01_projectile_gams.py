import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pygam import s, ExpectileGAM

np.random.seed(0)
n = 100
price  = np.sort(np.random.exponential(scale=100, size=n))
quantity = 1000 -5*price + np.random.normal(loc=0, scale=50, size=n)
quantity = quantity.clip(min=0)

n_outliers = 10
outlier_prices = np.random.uniform(5, 50, n_outliers)
outlier_quantity = 1100 + np.random.normal(loc=0, scale=50, size=n_outliers)
price = np.concatenate([price, outlier_prices])
quantity = np.concatenate([quantity, outlier_quantity])

outlier_prices = np.random.uniform(51, 100, n_outliers)
outlier_quantity = 1100 + np.random.normal(loc=0, scale=50, size=n_outliers)
price = np.concatenate([price, outlier_prices])
quantity = np.concatenate([quantity, outlier_quantity])

df = pd.DataFrame({'price':price, 'quantity': quantity})

X = df[['price']]
y = df['quantity']

quantiles = [0.025, 0.5, 0.975]
gam_results = {}

for q in quantiles:
    gam = ExpectileGAM(s(0), expectile=q)
    gam.fit(X,y)
    gam_results[q] = gam

gam_results

plt.figure(figsize=(10,6))
plt.scatter(df['price'], df['quantity'], alpha=0.5, label='Data Points')

XX = np.linspace(df['price'].min(), df['price'].max(), 1000).reshape(-1,1)

for q, gam in gam_results.items():
    plt.plot(XX, gam.predict(XX), label=f'{int(q*100)}th quantile gan')

plt.xlabel('price')
plt.ylabel('quantity demanded')
plt.title('quantile gams')
plt.legend()
plt.tight_layout()
plt.show()

