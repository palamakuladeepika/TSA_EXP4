# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 

### AIM:
To implement ARMA model in python.

### ALGORITHM:

**Step 1:** Import necessary libraries.

**Step 2:** Set up matplotlib settings for figure size.

**Step 3:** Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.

**Step 4:** Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.

**Step 5:** Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000 data points using the ArmaProcess class. Plot the generated time series and set the title and x-axis limits.

**Step 6:** Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.

### PROGRAM:
```
Developed By: Palamakula Deepika
Reg. No: 212221240035
```
```python
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
%matplotlib inline
plt.rcParams['figure.figsize'] = [10, 7.5]
train = pd.read_csv('Ex4.csv')
train['date'] = pd.to_datetime(train['date'], format='%d-%m-%Y')
train['Year'] = train['date'].dt.year
train['Values'] = train['open'].values.round().astype(int)
train.head()
ar1 = train['Year'].values.reshape(-1, 1)
ma1 = train['Values'].values.reshape(-1, 1)
ARMA_1 = ArmaProcess(ar1,ma1).generate_sample(nsample = 1000)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 200])
plt.show()
plot_acf(ARMA_1)
plot_pacf(ARMA_1)
ar2 = np.array([1, 0.33, 0.5])
ma2 = np.array([1, 0.9, 0.3])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=10000)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 50])
plt.show()
plot_acf(ARMA_2)
plot_pacf(ARMA_2)
```

### OUTPUT:
#### SIMULATED ARMA(1,1) PROCESS:

![image](https://github.com/Pavan-Gv/TSA_EXP4/assets/94827772/761e9e19-baa6-4658-a3ca-fccfe87fb97d)


**Partial Autocorrelation**

![image](https://github.com/Pavan-Gv/TSA_EXP4/assets/94827772/00759ffb-8500-4d29-9479-8dc1a0b37394)


**Autocorrelation**

![image](https://github.com/Pavan-Gv/TSA_EXP4/assets/94827772/b19e9bdb-206e-4e09-95f8-676205db613a)


#### SIMULATED ARMA(2,2) PROCESS:

![image](https://github.com/Pavan-Gv/TSA_EXP4/assets/94827772/446abf5f-6e59-4e82-820a-efdab936b8b5)


**Partial Autocorrelation**

![image](https://github.com/Pavan-Gv/TSA_EXP4/assets/94827772/0cdc608b-c714-49bb-9038-1c067e4cb6f8)


**Autocorrelation**

![image](https://github.com/Pavan-Gv/TSA_EXP4/assets/94827772/3941ad62-c870-4fcc-8815-bcd09786a3c0)


### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
