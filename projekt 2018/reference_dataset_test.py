import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from ts_regress import ts_regress, ts_regress_eval, test_to_Supervised, tsr_stack_indep
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score

# DEFINICIJE FUNKCIJ
# preverjanje stacionarnosti
def test_stationarity(timeSeries, window, label, plot):
    # @param timeSeries - casovna vrsta za testiranje, lahko je tipa numpy array
    # @param window - dolzina okna za izracun povprecja in std odklona
    # @param label - naslov grafa
    # @param plot - ce je true bo graf izrisan

    # pretvoba numpy arraya v pandas data frame
    df = pd.DataFrame(timeSeries)

    # tekoce povprecje in standardni odklon
    rolMean = pd.rolling_mean(df, window=window)
    rolStd = pd.rolling_std(df, window=window)

    # prikaz statistike
    if plot:
        plt.figure()
        plt.plot(df, color='blue', label='Original')
        plt.plot(rolMean, color='red', label='Rolling Mean')
        plt.plot(rolStd, color='black', label='Rolling Std')

        plt.legend(loc='best')
        plt.title('tekoce povprecje in standardni odklon:' + label)
        plt.show()


    #Perform Dickey-Fuller test:
    print('\nResults of Dickey-Fuller Test:')
    dftest = adfuller(df.squeeze(), autolag='AIC')
    print(np.ndim(dftest))
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


# BRANJE PODATKOV
data = pd.read_csv('AirPassengers.csv')
print(data.head())
print('\n Data Types:')
print(data.dtypes)

dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv('AirPassengers.csv', parse_dates=True, index_col='Month', date_parser=dateparse)
print(data.head())

data.index
ts = data['#Passengers']
ts.head(10)
ts = data.values
print(ts[0:10, :])
plt.figure()
plt.plot(ts)
plt.show()


#################################### NASTAVLJANJE MODELA ###############################################################
# dolocimo, koliko vzorcev nazaj upostevamo
nBack = 5

# dolocimo velikost podatkovnega seta
dataSetLen = 144

# dolocimo, kaksno dolzino podatkovnega seta namenimo ucni mnozici
trainDataLen = 100

# True ce zelimo pred analizo prikazati podatke
showData = False

# True SAMO ce napovedujemo iz premera zenice, nekoliko izboljsa R2
thresholdFeatures = True

# dolocimo vhodne znacilke (X) in label (Y)
# sestavimo podatkovni set, prvih 10 vzorcev izpustimo zaradi outlierjev
X = np.zeros(shape=[144, 1])
X[:, :] = ts
Y = ts

print("data shape:")
print(X.shape)
print(Y.shape)

print(X[0:10])

# STACIONARNOST CASOVNE VRSTE



# test pretvorbe v nadzorovano ucenje
#a = np.array([1, 2, 3, 4, 5, 6])
#b = np.array([11, 12, 13, 14, 15, 16])
#lag = 3
#print("test pretvorbe v nadzorovano ucenje")
#print(tsr_stack_indep(b, a, [lag])[1])
#print(test_to_Supervised(a, b, lag)[0])


def compute_r_squared(actual, predicted):
    return r2_score(actual, predicted)


# DELITEV NA UCNO IN TESTNO MNOZICO
X_train, X_test = X[0:trainDataLen, :], X[trainDataLen:, :]
y_train, y_test = Y[0:trainDataLen], Y[trainDataLen:]
print("train and test dataset shape: ")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# UCENJE MODELA
R2User = -5  # R^2 za trenutnega uporabnika
R2UserReg = 0
R2UserAdj = 0

for n in range(1, nBack):
    mod_ft = ts_regress(y_train, X_train, [n])
    R2Curr = r2_score(y_train[-mod_ft.fittedvalues.shape[0]:], mod_ft.fittedvalues)  # sklearn R2 da pravi rezultat
    # R2Curr = mod_ft.rsquared_adj
    if R2Curr > R2User:
        R2User = R2Curr
        R2UserAdj = mod_ft.rsquared_adj
        R2UserReg = mod_ft.rsquared
        nUser = [n]

# izpisi najvecji R2
print('max R^2: ', R2User, 'for n: ', nUser[0] + 1)
print('rsquared_adj:', R2UserAdj, 'rsquared:', R2UserReg, 'sklearn r2:', R2User)

# VALIDACIJA MODELA
mod_ft = ts_regress(y_test, X_test, nUser)

yRef = y_test[nBack:]
yPredMan = np.zeros(y_test.shape[0] - nBack)
print(yRef.shape, yPredMan.shape)
print(yPredMan.shape)
modelParams = np.flip(mod_ft.params, axis=0)  # parametre modela vrne v vrstnem redu: B0, B1, ...
print("model parameters: ", mod_ft.params)

for i in range(0, yPredMan.shape[0]):
    #predData = X_test[i:i+nUser[0]+1, :]
    predData = X_test[i:i + nUser[0], :]  # ce odrezemo stolpec trenutnih vrednosti znacilke
    yPredMan[i] = np.dot(modelParams, predData)

r2Man = compute_r_squared(yRef[0:(-nBack+1)], yPredMan[nBack-1:])  # popravimo zamike setov

plt.figure()
plt.plot(yRef, label='actual')
plt.plot(yPredMan[nBack-2:], label='predicted')
plt.legend(loc='best')
plt.show()
