import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from ts_regress import ts_regress, test_to_Supervised
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import r2_score

# POROCILO
# nasa zelja je, da napovemo tezko merljive podatke (valence, arousal) iz lahko merljivih (usta, zenica)
# ust nismo upostevali ker naj bi bile meritve diskretne
#
# 0. predprocesiranje zenice (rezultat je polmer zenice in ne sme biti <0) - nadomestimo z ustrezno interpolacijo
# 1. poskus z drugim datasetom (air passengers)
# 2. iskanje neizpolnjenih pogojev, zaradi katerih je modeliranje neuspesno
#       kandidati:
#        - napake v podatkih: manjkae na upanje in varianco, (ce zadeva ni stacionarna glede na matematicno upanje
#                                                           (za varianco resitve v bistvu ni) napovedujemo diferenco )
#                                               -> vrsto diferenciramo enkrat do dvakrat in preverimo novo stacionarnost
#        - sezone je treba odstranit
# *samopodobnosti nima smisla upostevati dokler ARIMA ne da vsaj priblizno ok rezultata
# 3. poskus z drugim modelom (LSTM) ce bo cas


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
    #print(np.ndim(dftest))
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)


def test_stationarity_dataset(features, label, window, title):
    # @brief test stacionarnosti za celoten set podatkov, trenutno deluje samo za eno znacilko

    figFeat, axesFeat = plt.subplots(3, 2)
    figLab, axesLab = plt.subplots(3, 2)

    figFeat.suptitle('feature, ' + title + ' window size: ' + str(window))
    figLab.suptitle('label, ' + title + ' window size: ' + str(window))

    for user in range(0, features.shape[0]):

        dfFeat = pd.DataFrame(features[user, :, 0])
        dfLab = pd.DataFrame(label[user, :])

        # nova verzija pandas vpeljuje drugacen klic metod
        rolMeanFeat = pd.rolling_mean(dfFeat, window=window)
        rolMeanLab = pd.rolling_mean(dfLab, window=window)

        rolStdFeat = pd.rolling_var(dfFeat, window=window)
        rolStdLab = pd.rolling_var(dfLab, window=window)

        # vizualizacija
        idxy = user % 3
        idxx = int(user/3)

        axesFeat[idxy, idxx].plot(dfFeat, label='data')
        axesFeat[idxy, idxx].plot(rolMeanFeat, label='rolling mean')
        axesFeat[idxy, idxx].plot(rolStdFeat, label='rolling variance')
        axesFeat[idxy, idxx].legend(loc='best')
        axesFeat[idxy, idxx].set_title('Feature for user {}'.format(uIDs[user]))

        axesLab[idxy, idxx].plot(dfLab, label='data')
        axesLab[idxy, idxx].plot(rolMeanLab, label='rolling mean')
        axesLab[idxy, idxx].plot(rolStdLab, label='rolling variance')
        axesLab[idxy, idxx].legend(loc='best')
        axesLab[idxy, idxx].set_title('Label for user {}'.format(uIDs[user]))

        # augumented Dickey-Fuller test
        print('\nFeature ADF test results ' + title + ' for user {}'.format(uIDs[user]))
        dftest = adfuller(dfFeat.squeeze(), autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)

        print('\nLabel ADF test results ' + title + ' for user {}'.format(uIDs[user]))
        dftest = adfuller(dfLab.squeeze(), autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value
        print(dfoutput)


def threshold_data(data, threshold):
    # @param data - numpy array [users, values, 1]
    for user in range(0, data.shape[0]):
        i = 1
        while i < data.shape[1] - 1:
            if data[user, i, :] < threshold:

                intStart = data[user, i - 1, 0]
                intStartIdx = i-1
                j = 0
                if data[user, i + 1, :] < threshold:
                    j = i + 1
                    while data[user, j, :] < threshold:
                        j += 1

                    intEnd = data[user, j, 0]
                    intEndIdx = j
                else:
                    intEnd = data[user, i + 1, 0]
                    intEndIdx = i+1

                #print(intStart, intEnd)
                #print(intStartIdx, intEndIdx)

                # replace the values below threshold
                for idx in range(intStartIdx + 1, intEndIdx):
                    data[user, idx, 0] = float(intEnd + intStart)/2

                i = intEndIdx

            i += 1

    return data


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
# print(uDataSize)

#################################### NASTAVLJANJE MODELA ###############################################################
# dolocimo, koliko vzorcev nazaj upostevamo
nBack = 12

# dolocimo velikost podatkovnega seta
dataSetLen = maxDataSize

# dolocimo, kaksno dolzino podatkovnega seta namenimo ucni mnozici
trainDataLen = 1500

# True ce zelimo pred analizo prikazati podatke
showData = False

# True SAMO ce napovedujemo iz premera zenice, nekoliko izboljsa R2
thresholdFeatures = True

# True ce uporabimo diferenciranje prvega reda
differencing = True

# velikost drsecega okna za racunanje tekocega povprecja, variance, ...
slidingWindowSize = 20

# dolocimo vhodne znacilke (X) in label (Y)
# sestavimo podatkovni set, prvih 10 vzorcev izpustimo zaradi outlierjev
X = np.zeros(shape=[len(uIDs), dataSetLen - 10, 1])
Y = np.zeros(shape=[len(uIDs), dataSetLen - 10])
for idx, val in enumerate(uIDs):
    print(val)
    X[idx] = np.array([reg_leftPupilDiameter[val][10:dataSetLen]]).T
    Y[idx] = np.array([reg_valence[val][10:dataSetLen]])

########################################################################################################################

# PREDPROCESIRANJE PODATKOV - SAMO ZA ZENICO
pupilThreshold = 1
if thresholdFeatures:
    X = threshold_data(X, pupilThreshold)

print("data shape:")
print(X.shape)
print(Y.shape)
print(X[0, 0:10])
print(Y[0, 0:10])

# STACIONARNOST CASOVNE VRSTE
test_stationarity_dataset(X, Y, slidingWindowSize, 'before differencing')

# DIFERENCIRANJE PODATKOV
if differencing:

    #X = X[:, 1:, :]
    for user in range(0, Y.shape[0]):
        for i in range(0, Y.shape[1] - 1):
            Y[user, i] = Y[user, i+1] - Y[user, i]
            X[user, i, 0] = X[user, i+1, 0] - X[user, i, 0]

    Y = Y[:, :-1]
    X = X[:, :-1, :]

    print("data shape after differencing: ")
    print(X.shape)
    print(Y.shape)

    # ponovni test stacionarnosti
    test_stationarity_dataset(X, Y, slidingWindowSize, 'after differencing')

plt.show()

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


def compute_r_squared(actual, predicted):
    return r2_score(actual, predicted)


# DELITEV NA UCNO IN TESTNO MNOZICO
X_train, X_test = X[:, 0:trainDataLen, :], X[:, trainDataLen:, :]
y_train, y_test = Y[:, 0:trainDataLen], Y[:, trainDataLen:]
print("train and test dataset shape: ")
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# UCENJE MODELA
# izvedemo ucenje za vsakega uporabnika
# regresija z metodo najmanjsih kvadratov (statsmodels.api.OLS)
# na podlagi trenutne in preteklih vrednosti znacilke napovedujemo trenutno vrednost labela
for idx, uID in enumerate(uIDs):

    print("computation running for user", uID)

    R2Series = np.empty(nBack * nBack * nBack)
    R2User = -5  # R^2 za trenutnega uporabnika
    R2UserReg = 0
    R2UserAdj = 0
    nUser = [0, 0, 0]

    # delitev na ucno in testno mnozico
    y = y_train[idx]
    x = X_train[idx]

    for n in range(1, nBack):
        mod_ft = ts_regress(y, x, [n])
        R2Curr = r2_score(y[-mod_ft.fittedvalues.shape[0]:], mod_ft.fittedvalues)  # sklearn R2 da pravi rezultat
        #R2Curr = mod_ft.rsquared_adj
        if R2Curr > R2User:
            R2User = R2Curr
            R2UserAdj = mod_ft.rsquared_adj
            R2UserReg = mod_ft.rsquared
            nUser = [n]

    # izpisi najvecji R2
    print('max R^2: ', R2User, 'for n: ', nUser[0] + 1)
    print('rsquared_adj:', R2UserAdj, 'rsquared:', R2UserReg, 'sklearn r2:', R2User)

    # VALIDACIJA MODELA
    # pokazi fit za najboljsi R2
    mod_ft = ts_regress(y, x, nUser)

    # pofitane vrednosti so tipa numpy array
    # problem: toliko vzorcev kolikor najvec gledamo nazaj, toliko krajsi fit dobimo

    # 1. metoda: fittedvalues -> neuporabno, saj uporabi podatke ucne mnozice
    yPred = mod_ft.fittedvalues
    yRef = y_test[idx, nBack:]
    fig, (ax2, ax3) = plt.subplots(2, 1)
    # ax1.plot(yPred, label='predicted')
    # ax1.plot(yRef, label='actual')
    # ax1.legend(loc='best')
    # ax1.set_title("Model fit to training set for user"+str(uID))

    # 2. metoda: koeficiente modela se da dobiti in predikcije racunati rocno
    # print(mod_ft.params)
    # print(X.shape)

    yPredMan = np.zeros(y_test.shape[1] - nBack)
    #print(yPredMan.shape)
    modelParams = np.flip(mod_ft.params, axis=0)  # parametre modela vrne v vrstnem redu: B0, B1, ...
    #print("model parameters: ", mod_ft.params)

    for i in range(0, yPredMan.shape[0]):
        predData = X_test[idx, i:i+nUser[0]+1, :]
        #predData = X[idx, i:i + nUser[0], :]  # ce odrezemo stolpec trenutnih vrednosti znacilke
        yPredMan[i] = np.dot(modelParams, predData)


    r2Man = compute_r_squared(yRef[0:(-nBack+1)], yPredMan[nBack-1:])  # popravimo zamike setov

    ax2.plot(yRef[0:(-nBack+1)], label='actual')
    ax2.plot(yPredMan[nBack-1:], label='fitted')
    ax2.legend(loc='best')
    #ax2.set_title("Manual model predict for user " + str(uID) + ", R2 score %.4f" % r2Man)
    ax2.set_title("Model results for user " + str(uID))

    # 3. metoda: predikcija z uporabo metode predict
    testData, _ = test_to_Supervised(X_test[idx], y_test[idx], nUser[0])
    yPredAuto = mod_ft.predict(testData)  # Call self.model.predict with self.params as the first argument.

    ax3.plot(yRef, label='actual')
    ax3.plot(yPredAuto[nBack-1:], label='predicted')
    ax3.legend(loc='best')
    ax3.set_title("Model predict for user " + str(uID))

    print("oblike podatkov: ")
    print("params: ", mod_ft.params.shape)
    print("yRef", yRef.shape)
    print("fitted values: ", yPredAuto.shape)

plt.show()
