import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from ts_regress import ts_regress, ts_regress_eval
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as smAPI

# TODO list
# normalizacija casovnih vrst (naj bodo cim bolj stacionarne) - kot v laboratorijski vaji
# vizualizacija podatkov -> vec grafov v eno sliko
# delta pupil


# DEFINICIJE FUNKCIJ
# preverjanje stacionarnosti
def test_stationarity(timeSeries, window, label, plot):

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
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(df.squeeze(), autolag='AIC')
    print(np.ndim(dftest))
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print (dfoutput)


# SPREMENLJIVKE
uIDs = [10, 19, 20, 27, 34]  # IDji uporabnikov
userCount = len(uIDs)

# read pickle python format to dictionaries
# pickle - serializes a python object into stream of bytes
# vsak dictionary ima pet polj - za vsakega uporabnika eno
reg_arousal = pickle.load(open('arousal.pckl', "rb"))               # vzburjenost/vznemirjenje
reg_valence = pickle.load(open('valence.pckl', "rb"))               # valenca/prisotnost
reg_times = pickle.load(open('times.pckl', "rb"))                   # casovne znamke
reg_emo_mouth = pickle.load(open('mouth.pckl', "rb"))               # usta
reg_leftPupilDiameter = pickle.load(open('Lpupil.pckl', "rb"))      # premer leve zenice  #IDENTICNA
reg_rightPupilDiameter = pickle.load(open('Rpupil.pckl', "rb"))     # premer desne zenice #IDENTICNA

# dictionary se bere po kljucu
# vsak uporabnik ima razlicno dolzino casovne vrste
uDataSize = [len(reg_arousal[10]), len(reg_arousal[19]),
             len(reg_arousal[20]), len(reg_arousal[27]),
             len(reg_arousal[34])]
maxDataSize = min(uDataSize)
print(uDataSize)


# NASTAVLJANJE MODELA
# dolocimo, koliko vzorcev nazaj upostevamo
nBack = 5

# dolocimo velikost podatkovnega seta
dataSetLen = 50

# dolocimo, katero znacilko iscemo (label)
Y = reg_arousal

print 'primerjava zenic'
print np.sum(np.subtract(reg_leftPupilDiameter[19][0:dataSetLen], reg_rightPupilDiameter[19][0:dataSetLen]))

# dolocimo vhodne znacilke
# sestavimo podatkovni set
X = np.zeros(shape=[len(uIDs), dataSetLen, 3])
for idx, val in enumerate(uIDs):
    print val
    X[idx] = np.array([reg_leftPupilDiameter[val][0:dataSetLen],
                       reg_rightPupilDiameter[val][0:dataSetLen],
                       reg_emo_mouth[val][0:dataSetLen]]).T

print("data shape:")
print(X.shape)

# STACIONARNOST CASOVNE VRSTE
test_stationarity(X.T[0, :, 0], 12, 'usta', 0)
test_stationarity(X.T[0, :, 1], 12, 'desna zenica', 0)
test_stationarity(X.T[0, :, 2], 12, 'leva zenica', 0)

# prikaz labela
#plt.figure()
#plt.plot(Y[19][0:dataSetLen], color='blue', label='Original')
#plt.legend(loc='best')
#plt.title('Label')
#plt.show()

# UCENJE MODELA
R2SeriesAllUsers = np.empty(shape=[userCount, nBack * nBack * nBack])

# izvedemo ucenje za vsakega uporabnika
# regresija z metodo najmanjsih kvadratov (statsmodels.api.OLS)
for idx, uID in enumerate(uIDs):

    print "computation running for user", uID

    y = Y[uID][0:dataSetLen]

    #print('data: ')
    #print(y)
    #print(X[idx])
    #print np.subtract(X[idx, :, 0], X[idx, :, 1])

    R2Series = np.empty(nBack * nBack * nBack)
    R2User = 0  # R^2 za trenutnega uporabnika
    nUser = [0, 0, 0]

    # regresijo izvajamo za vse mozne kombinacije propagacij
    cnt = 0
    for n1 in range(1, nBack + 1):
        for n2 in range(1, nBack + 1):
            for n3 in range(1, nBack + 1):
                p_lags = [n1, n2, n3]
                mod_ft = ts_regress(y, X[idx], p_lags)     # metoda najmanjsih kvadratov, tip: statsmodels.regression.linear_model.RegressionResults
                R2Series[cnt] = mod_ft.rsquared_adj

                if mod_ft.rsquared_adj > R2User:
                    R2User = mod_ft.rsquared_adj
                    nUser = [n1, n2, n3]
                cnt += 1

    R2SeriesAllUsers[idx] = R2Series

    # izpisi najvecji R2
    print 'max R^2: ', R2User, 'for n: ', nUser

    # pokazi fit za najboljsi R2
    plt.figure()
    mod_ft = ts_regress(y, X[idx], nUser)
    #print type(mod_ft)
    fig, ax = plt.subplots()
    smAPI.graphics.plot_fit(mod_ft, 0, ax=ax)
    #ax.set_ylabel()
    #ax.set_xlabel()
    ax.set_title("User " + str(uID))
    plt.show()
    print mod_ft.fittedvalues.shape  # pofitane vrednosti so tipa numpy array
                                     # problem: toliko vzorcev kolikor najvec gledamo nazaj, toliko krajsi fit dobimo



# dolocimo uspesnost prileganja modela - R^2
# pove, koliko bolje se nas model prilega glede na povprecenje vrednosti (konstantni model)

# PRIKAZ REZULTATOV
print "\nRESULTS"
print(R2SeriesAllUsers.min())
print(R2SeriesAllUsers.max())
#print(R2[0])

# R^2 je merilo, kako dobro se model prilega podatkom. Ce je enak 1 to pomeni da se model popolnoma prilega
# podatkom. Ce je rezultat izven obmocja 0-1 to pomeni, da je model verjetno napacen, popolnoma narobe nastavljen,
# imamo prevec znacilk ipd...




