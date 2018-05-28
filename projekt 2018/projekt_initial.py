import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from ts_regress import ts_regress, ts_regress_eval, test_to_Supervised
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score
import statsmodels.api as smAPI

# TODO list
# normalizacija casovnih vrst (naj bodo cim bolj stacionarne) - kot v laboratorijski vaji
# vizualizacija podatkov -> vec grafov v eno sliko
# delta pupil


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


# SPREMENLJIVKE
uIDs = [10, 19, 20, 27, 34]  # IDji uporabnikov
userCount = len(uIDs)

# vzorcna frekvenca za podatke je bila 40Hz (oziroma 10Hz downsamplano)
# usta naj bi bila diskretna (NE uporabljaj ust), poskusimo zenico in valence
# smiselno bi bilo poskusiti se z vecjimi zamiki
# napovedi se delajo SAMO na podlagi tistega, kar je v X mnozici (AR model)
# napovedi se delajo za trenutni y vzorec na podlagi trenutnega in preteklih X vzorcev

#TODO
# preveri ce se zamiki vrste delajo ok
# na podlagi koeficientov izracunaj napovedi in poglej, ce se ujemajo z napovedmi .fittedvalues
# preveri, v kaksnem vrstnem redu mod_ft vrne koeficiente

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
#print(uDataSize)

#################################### NASTAVLJANJE MODELA ###############################################################
# dolocimo, koliko vzorcev nazaj upostevamo
nBack = 5

# dolocimo velikost podatkovnega seta
dataSetLen = maxDataSize

# dolocimo, kaksno dolzino podatkovnega seta namenimo ucni mnozici
trainDataLen = 2000

# True ce zelimo pred analizo prikazati podatke
showData = True

# dolocimo vhodne znacilke (X) in label (Y)
# sestavimo podatkovni set
X = np.zeros(shape=[len(uIDs), dataSetLen, 1])
Y = np.zeros(shape=[len(uIDs), dataSetLen])
for idx, val in enumerate(uIDs):
    print(val)
    X[idx] = np.array([#reg_leftPupilDiameter[val][0:dataSetLen],
                       #reg_rightPupilDiameter[val][0:dataSetLen],
                       reg_leftPupilDiameter[val][0:dataSetLen]]).T
    Y[idx] = np.array([reg_valence[val][0:dataSetLen]])


########################################################################################################################

print("data shape:")
print(X.shape)
print(Y.shape)

# STACIONARNOST CASOVNE VRSTE
#test_stationarity(X.T[0, :, 0], 12, 'usta', 0)
#test_stationarity(X.T[0, :, 1], 12, 'desna zenica', 0)
#test_stationarity(X.T[0, :, 2], 12, 'leva zenica', 0))
#test_stationarity(Y[0], 12, 'label user 19', 1)

# prikaz labela
#plt.figure()
#plt.plot(X[0], color='blue', label='Original')
#plt.legend(loc='best')
#plt.title('Label for user 10')
#plt.show()

# PRIKAZ PODATKOV
if showData:
    fig, axes = plt.subplots(5, 2)

    axes[0, 0].plot(X[0], label='feature')
    axes[0, 0].legend(loc='best')
    axes[0, 0].set_title("user 10")

    axes[0, 1].plot(Y[0], label='label')
    axes[0, 1].legend(loc='best')
    axes[0, 1].set_title("user 10")

    axes[1, 0].plot(X[1], label='feature')
    axes[1, 0].legend(loc='best')
    axes[1, 0].set_title("user 19")

    axes[1, 1].plot(Y[1], label='label')
    axes[1, 1].legend(loc='best')
    axes[1, 1].set_title("user 19")

    axes[2, 0].plot(X[2], label='feature')
    axes[2, 0].legend(loc='best')
    axes[2, 0].set_title("user 20")

    axes[2, 1].plot(Y[2], label='label')
    axes[2, 1].legend(loc='best')
    axes[2, 1].set_title("user 20")

    axes[3, 0].plot(X[3], label='feature')
    axes[3, 0].legend(loc='best')
    axes[3, 0].set_title("user 27")

    axes[3, 1].plot(Y[3], label='label')
    axes[3, 1].legend(loc='best')
    axes[3, 1].set_title("user 27")

    axes[4, 0].plot(X[4], label='feature')
    axes[4, 0].legend(loc='best')
    axes[4, 0].set_title("user 34")

    axes[4, 1].plot(Y[4], label='label')
    axes[4, 1].legend(loc='best')
    axes[4, 1].set_title("user 34")

    plt.show()


# DELITEV NA UCNO IN TESTNO MNOZICO
X_train, X_test = X[:, 0:trainDataLen, :], X[:, trainDataLen:, :]
y_train, y_test = Y[:, 0:trainDataLen], Y[:, trainDataLen:]
print("train and test dataset shape: ")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# UCENJE MODELA
# izvedemo ucenje za vsakega uporabnika
# regresija z metodo najmanjsih kvadratov (statsmodels.api.OLS)
# na podlagi trenutne in preteklih vrednosti featurejev napovedujemo trenutno vrednost labela
for idx, uID in enumerate(uIDs):

    print("computation running for user", uID)

    y = y_train[idx]

    R2Series = np.empty(nBack * nBack * nBack)
    R2User = -5  # R^2 za trenutnega uporabnika
    R2UserReg = 0
    R2UserAdj = 0
    nUser = [0, 0, 0]

    # delitev na ucno in testno mnozico


    # regresijo izvajamo za vse mozne kombinacije propagacij
    # cnt = 0
    #for n1 in range(1, nBack + 1):
    #    for n2 in range(1, nBack + 1):
    #        for n3 in range(1, nBack + 1):
    #            p_lags = [n1, n2, n3]
    #            mod_ft = ts_regress(y, X[idx], p_lags)     # metoda najmanjsih kvadratov, tip: statsmodels.regression.linear_model.RegressionResults
    #            #R2Series[cnt] = mod_ft.rsquared_adj

                #if mod_ft.rsquared_adj > R2User:
                #    R2User = mod_ft.rsquared_adj
                #    nUser = [n1, n2, n3]

                # izracun R2 za dano iteracijo
    #            R2Curr = r2_score(y[-mod_ft.fittedvalues.shape[0]:], mod_ft.fittedvalues)
    #            if R2Curr > R2User:
    #                R2User = R2Curr
    #                nUser = [n1, n2, n3]

    #            cnt += 1

    for n in range(1, nBack):
        mod_ft = ts_regress(y, X_train[idx], [n])
        R2Curr = r2_score(y[-mod_ft.fittedvalues.shape[0]:], mod_ft.fittedvalues)
        #R2Curr = mod_ft.rsquared_adj
        if R2Curr > R2User:
            R2User = R2Curr
            R2UserAdj = mod_ft.rsquared_adj
            R2UserReg = mod_ft.rsquared
            nUser = [n, 0, 0]

    # izpisi najvecji R2
    print('max R^2: ', R2User, 'for n: ', nUser)
    print('rsquared_adj:', R2UserAdj, 'rsquared:', R2UserReg, 'sklearn r2:', R2User)

    # pokazi fit za najboljsi R2
    mod_ft = ts_regress(y, X_train[idx], nUser) #prej: nUser
    #print(mod_ft.fittedvalues.shape)  # pofitane vrednosti so tipa numpy array
                                      # problem: toliko vzorcev kolikor najvec gledamo nazaj, toliko krajsi fit dobimo

    # koeficiente modela vzeti iz mod_ft
    #print(mod_ft.params)

    #TODO UCENJE NA TESTNEM DATASETU
    # prikaz dejanskih in napovedanih podatkov
    yPred = mod_ft.fittedvalues
    yRefTrain = y[-yPred.shape[0]:]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    ax1.plot(yPred, label='predicted')
    ax1.plot(yRefTrain, label='actual')
    ax1.legend(loc='best')
    ax1.set_title("Model fit for user"+str(uID))

    # rocni izracun predvidenih vrednosti
    #print(X.shape)
    yPrMan = np.zeros(X.shape[1] - nBack)
    #modelParams = mod_ft.params
    modelParams = np.flip(mod_ft.params, axis=0)  # parametre modela ocitno vrne v vrstnem redu: B0, B1, ...
    print("model parameters: ", modelParams, mod_ft.params)

    for i in range(0, yPrMan.shape[0]):
        predData = X[idx, i:i+nUser[0]+1, :]
        yPrMan[i] = np.dot(modelParams, predData)

    ax2.plot(yPrMan, label='predicted')
    ax2.plot(yRefTrain, label='actual')
    ax2.legend(loc='best')
    ax2.set_title("Manual model predict for user" + str(uID))

    # predikcija z uporabo metode predict
    testData, _ = test_to_Supervised(X[idx], y, nUser[0])
    yPr = mod_ft.predict(testData)  # Call self.model.predict with self.params as the first argument.

    ax3.plot(yPr, label='predicted')
    ax3.plot(yRefTrain, label='actual')
    ax3.legend(loc='best')
    ax3.set_title("Model predict for user" + str(uID))

    #fig, ax = plt.subplots()
    # This creates one graph with the scatterplot of observed values compared to fitted values.
    #smAPI.graphics.plot_fit(mod_ft, 0, ax=ax)
    #ax.set_ylabel('y')
    #ax.set_xlabel('x')
    #ax.set_title('Model fit for user'+str(uID))

plt.show()

# dolocimo uspesnost prileganja modela - R^2
# pove, koliko bolje se nas model prilega glede na povprecenje vrednosti (konstantni model)
# R^2 je merilo, kako dobro se model prilega podatkom. Ce je enak 1 to pomeni da se model popolnoma prilega
# podatkom. Ce je rezultat izven obmocja 0-1 to pomeni, da je model verjetno napacen, popolnoma narobe nastavljen,
# imamo prevec znacilk ipd...
