import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from ts_regress import ts_regress, ts_regress_eval
from statsmodels.tsa.stattools import adfuller
#TO SEM DODAL !!!!!!!!!!!!!!!!!!
from statsmodels.tsa.seasonal import seasonal_decompose
#
from sklearn.metrics import r2_score
import statsmodels.api as smAPI

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


print("Done!")


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

# PREVERI DOLZINO VSEH SETOV PODATKOV!!!!!!!!!!!!!!!!!!!!!!!!!!!!!###############!!!!!!!########!!!!!!!
# dictionary se bere po kljucu (ID uporabnika)
# vsak uporabnik ima razlicno dolzino casovne vrste
uDataSize = [len(reg_arousal[10]), len(reg_arousal[19]),
             len(reg_arousal[20]), len(reg_arousal[27]),
             len(reg_arousal[34])]
maxDataSize = min(uDataSize)
#print(uDataSize)
#print(maxDataSize)

# dolocimo velikost podatkovnega seta
dataSetLen = 1000

# sestavimo podatkovni set
pupil = np.zeros(shape=[len(uIDs), dataSetLen, 1])
mouth = np.zeros(shape=[len(uIDs), dataSetLen])
valence = np.zeros(shape=[len(uIDs), dataSetLen])
arousal = np.zeros(shape=[len(uIDs), dataSetLen])
for idx, val in enumerate(uIDs):
    pupil[idx] = np.array([reg_rightPupilDiameter[val][0:dataSetLen]]).T
    mouth[idx] = np.array([reg_emo_mouth[val][0:dataSetLen]])
    valence[idx] = np.array([reg_valence[val][0:dataSetLen]])
    arousal[idx] = np.array([reg_arousal[val][0:dataSetLen]])
    
print("Done!")


# tu dolocas za katerega uporabnika prikazujes, uporabi preslikavo naslovov
user = 4 # user [0,  1,  2,  3,  4]
         # ID   [10, 19, 20, 27, 34]
# STACIONARNOST ORIGINALNE CASOVNE VRSTE
test_stationarity(pupil[user], 12, 'zenica', 1)
test_stationarity(mouth[user], 12, 'usta', 1)
test_stationarity(valence[user], 12, 'valenca', 1)
test_stationarity(arousal[user], 12, 'arousal', 1)


### VSEM PODATKOM PRISTEJEMO (NAJMANJSO VREDNOST + 1), ZATO DA DELUJE LOGARITEM

# OPOMBA: ce veckrat pozenes to se np.amin(pupil) spremeni
        # to kar moras na koncu pristevat, izracunaj ze v zgornjem razdelku

#print(np.amax(pupil))
#print(np.amin(pupil))
pupil = pupil - np.amin(pupil) + 1 #to moras na koncu odsteti !!!
#print(np.amax(pupil))
#print(np.amin(pupil))
#print(type(pupil))

# print(np.amax(mouth))
# print(np.amin(mouth))
mouth = mouth - np.amin(mouth) + 1 #to moras na koncu odsteti !!!
# print(np.amax(mouth))
# print(np.amin(mouth))

# print(np.amax(valence))
# print(np.amin(valence))
valence = valence - np.amin(valence) + 1 #to moras na koncu odsteti !!!
# print(np.amax(valence))
# print(np.amin(valence))

# print(np.amax(arousal))
# print(np.amin(arousal))
arousal = arousal - np.amin(arousal) + 1 #to moras na koncu odsteti !!!
# print(np.amax(arousal))
# print(np.amin(arousal))


# TU IZBERI KATEREGA UPORABNIKA ZELIS
# za enega izmed user [0,  1,  2,  3,  4]
               # ID   [10, 19, 20, 27, 34]
user = 1

# TU NASTAVI KATERI SET PODATKOV HOCES OBDELAT
# pretvorba: numpy.ndarray ---> pandas.core.frame.DataFrame
pandas = pd.DataFrame(arousal[user]) 
#print(type(pandas))

# logaritmiranje podatkov
log_pandas = np.log(pandas)
#print(type(log_pandas))

# tekoce povprecje
window_length = 12
log_avg_pandas = pd.rolling_mean(log_pandas, window_length)

# razlika log - log_avg
diff_log_avg_pandas = log_pandas - log_avg_pandas

# izris
plt.figure()
plt.plot(log_pandas, color="purple", label="Log")
plt.plot(log_avg_pandas, color='green', label="Moving average")
plt.plot(diff_log_avg_pandas, color='black', label="Difference")

# testiranje stacionariziranosti diff_log_avg
diff_log_avg_pandas.dropna(inplace=True) # vse NaN spusti?????
test_stationarity(diff_log_avg_pandas, 12, 'diff_log_avg', 1)


# TU IZBERI KATEREGA UPORABNIKA ZELIS
# za enega izmed user [0,  1,  2,  3,  4]
               # ID   [10, 19, 20, 27, 34]
user = 1

# TU NASTAVI KATERI SET PODATKOV HOCES OBDELAT
# pretvorba: numpy.ndarray ---> pandas.core.frame.DataFrame
pandas = pd.DataFrame(arousal[user]) 
#print(type(pandas))

# logaritmiranje podatkov
log_pandas = np.log(pandas)
#print(type(log_pandas))

# exponential weighted moving average
window_length = 12
log_ewma_pandas = pd.ewma(log_pandas, window_length)

# razlika log - log_ewma
diff_log_ewma_pandas = log_pandas - log_ewma_pandas

# izris
plt.figure()
plt.plot(log_pandas, color="purple", label="Log")
plt.plot(log_ewma_pandas, color='green', label="EWMA")
plt.plot(diff_log_ewma_pandas, color='black', label="Difference")

# testiranje stacionariziranosti diff_log_ewma
diff_log_ewma_pandas.dropna(inplace=True) # vse NaN spusti?????
test_stationarity(diff_log_ewma_pandas, 12, 'diff_log_ewma', 1)


# TU IZBERI KATEREGA UPORABNIKA ZELIS
# za enega izmed user [0,  1,  2,  3,  4]
               # ID   [10, 19, 20, 27, 34]
user = 1

# TU NASTAVI KATERI SET PODATKOV HOCES OBDELAT
# logaritmiranje podatkov
log_pandas = np.log(valence[user])
#print(type(log_pandas))

# DEKOMPOZICIJA
decomposition = seasonal_decompose(log_pandas, freq=100)

#sedaj shranimo v posebne spremanljivke da lahko posebaj klicemo
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# odstranjevanje NaN in Inf vrednosti
#print(np.amax(residual))
#print(np.amin(residual))
residual = residual[np.isfinite(residual)]
residual = residual[~np.isnan(residual)]
#print(np.amax(residual))
#print(np.amin(residual))

#izris
plt.figure()
plt.subplot(411)
plt.plot(log_pandas, label='Original (log_pandas)')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

#sedaj testiramo kako so stacionarizirani "residual"
test_stationarity(residual, 12, 'Residual (DECOMPOSE)', 1)
