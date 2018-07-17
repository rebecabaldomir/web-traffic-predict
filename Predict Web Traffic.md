
# Time Series: Predict the Web Traffic

Dado os dados de tráfego da Web medidos em termos de sessões de usuários será necessário prever o número de sessões pros próximos 30 dias. Para isso, é disponibilizado os dados de 1133 dias a partir de 1 de Outubro de 2012.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from statsmodels.tsa.seasonal import seasonal_decompose
```


```python
data = pd.read_table('input/input01.txt',names=["sessions"])
data.drop([0],inplace=True) #primeira linha guarda o número de linhas que o arquivo tem
```


```python
data.plot(kind='line')
```




    <matplotlib.axes._subplots.AxesSubplot at 0xd2e17f0>




![png](output_4_1.png)


#### Decomposição

A decomposição é usada principalmente para análise de séries temporais e, como uma ferramenta de análise, pode ser usada para informar modelos de previsão sobre seu problema.

Ele fornece uma maneira estruturada de pensar sobre um problema de previsão de série temporal, geralmente em termos de complexidade de modelagem e especificamente em termos de como capturar melhor cada um desses componentes em um determinado modelo.


```python
data.index = pd.DatetimeIndex(freq='d', start='2012-08-01', periods=500)
result = seasonal_decompose(data, model='additive')
result.plot()
plt.show()
```


![png](output_7_0.png)


### Modelo

#### Cross-Validation


```python
arq = pd.read_table('input/input01.txt',names=["sessions"])
X = arq.index.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
```

#### Aprendizagem Supervisionada


```python
from sklearn.linear_model import LogisticRegression
regr = LogisticRegression()
regr.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)



#### Predição


```python
pred = regr.predict(X_test)
```

Fonte:  
https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/  
http://www.ulb.ac.be/di/map/gbonte/ftp/time_ser.pdf  
http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
