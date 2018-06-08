import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import tensorflow as tf
from tensorflow.python.client import device_lib
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras import backend as K

# TODO list
# vpliv zacetnega stanja mreze
# pretvorba v nadzorovano ucenje za multivariantne vrste
# uporaba drugacne zgradbe mreze

# preveri ce tensorflow uporablja GPU
print(device_lib.list_local_devices())

K.tensorflow_backend._get_available_gpus()

# nastavi koliko GPU naj se uporablja
configKeras = tf.ConfigProto()
configKeras.gpu_options.per_process_gpu_memory_fraction = 0.9
K.set_session(tf.Session(config=configKeras))

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
    print(dfoutput)


# SPREMENLJIVKE
uIDs = [10, 19, 20, 27, 34]  # IDji uporabnikov
userCount = len(uIDs)

# read pickle python format to dictionaries
# pickle - serializes a python object into stream of bytes
# vsak dictionary ima pet polj - za vsakega uporabnika eno
# NOTE - v python3 ima pickle drugacen encoding zato ni direktno kompatibilen z pyhton 2
reg_arousal = pickle.load(open('arousal.pckl', "rb"), encoding='latin1')               # vzburjenost/vznemirjenje
reg_valence = pickle.load(open('valence.pckl', "rb"), encoding='latin1')               # valenca/prisotnost
reg_times = pickle.load(open('times.pckl', "rb"), encoding='latin1')                   # casovne znamke
reg_emo_mouth = pickle.load(open('mouth.pckl', "rb"), encoding='latin1')               # usta
reg_leftPupilDiameter = pickle.load(open('Lpupil.pckl', "rb"), encoding='latin1')      # premer leve zenice  #IDENTICNA
reg_rightPupilDiameter = pickle.load(open('Rpupil.pckl', "rb"), encoding='latin1')     # premer desne zenice #IDENTICNA

# dictionary se bere po kljucu
# vsak uporabnik ima razlicno dolzino casovne vrste
uDataSize = [len(reg_arousal[10]), len(reg_arousal[19]),
             len(reg_arousal[20]), len(reg_arousal[27]),
             len(reg_arousal[34])]
maxDataSize = min(uDataSize)
print(uDataSize)


######################################## NASTAVLJANJE MODELA ###########################################################
# dolocimo, koliko vzorcev nazaj upostevamo
nBack = 3

# dolocimo velikost batcha pri ucenju (mora biti delitelj velikosti ucne mnozice)
#batchSize = 50 #nastavim ga na train_size za hitrejse ucenje

# dolocimo velikost podatkovnega seta
dataSetLen = maxDataSize

# dolocimo razmerje med ucno in testno mnozico
testTrainRatio = 0.3

# stevilo nevronov v lstm
nNeurons = 3

# kolikokrat naj se izvede ucenje modela
nEpoch = 1000

# dolocimo vhodne znacilke
# sestavimo podatkovni set
XAllUsers = np.zeros(shape=[len(uIDs), dataSetLen, 2])
for idx, val in enumerate(uIDs):
    print(val)
    XAllUsers[idx] = np.array([reg_emo_mouth[val][0:dataSetLen],
                               reg_valence[val][0:dataSetLen]]).T          # label


plt.figure()
plt.plot(XAllUsers[0, :, 1])
plt.title('label')
plt.show()

# pandas dataframe mora biti 2D -> delamo za vsakega uporabnika posebej
print(XAllUsers.shape)
XAllUsers = pd.DataFrame(XAllUsers[0, :, :])  # tukaj izberi indeks uporabnika
print("data shape:")
print(XAllUsers.shape)
print(XAllUsers.head())

########################################################################################################################

# PRETVORBA V NADZOROVANO UCENJE
# LSTM predvideva delitev na input (X) in output (Y)
# pretvorba po metodi drsecega okna
# polja NaN zaradi zamika nadomestimo z 0


def timeSeries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]    # ustvarimo zamaknjene stolpce
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)                          # zamenja NaN z 0
    return df


print("\nConverting time series to supervised learning")
# print(timeSeries_to_supervised(X, 2).head())
XAllUsers = timeSeries_to_supervised(XAllUsers, nBack)
print(XAllUsers.head())

# pretvorba v numpy array
Xnp = XAllUsers.values
print(Xnp.shape)

# DELITEV NA UCNO IN TESTNO MNOZICO
print("\nSplitting into training and testing data set")
#train, test = Xnp[0:-30], Xnp[-30:]
#trainSize = int(batchSize*(round(float(dataSetLen*(1 - testTrainRatio))/batchSize )))
trainSize = int(round(float(dataSetLen) * (1 - testTrainRatio)))
train = Xnp[0:trainSize]
test = Xnp[trainSize:]
print("train shape: ", train.shape, "test shape: ", test.shape)

# SKALIRANJE CASOVNE VRSTE
# aktivacijska funkcija v nevronski mrezi (privzveto tanh) zahteva ustezno obmocje vrednosti v podatkovnem setu
# koeficienta skaliranja naj bosta izracunana v ucni mnozici in upostevana tudi v testni, ter pri napovedih
# nazaj se vrnemo s klicem scaler.inverse_transform()


def scale_data(trainSet, testSet):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(trainSet)

    # pretvorba ucne mnozice
    trainSet = trainSet.reshape(trainSet.shape[0], train.shape[1])
    trainSetScaled = scaler.transform(trainSet)

    # pretvorba testne mnozice
    testSet = testSet.reshape(testSet.shape[0], testSet.shape[1])
    # print('oblika testnega: ', testSet.shape)
    testSetScaled = scaler.transform(testSet)
    return scaler, trainSetScaled, testSetScaled


def inverse_scale_data(scaler, data, value):
    row = [x for x in data] + [value]
    array = np.array(row)
    array = array.reshape(1, len(array))
    inverse = scaler.inverse_transform(array)
    return inverse[0, -1]


print("\nScaling the dataset")
MMscaler, trainScaled, testScaled = scale_data(train, test)

# MODEL
# LSTM (long short-term memory) je podvrsta RNN (recurrent neural network)
# z argumentom 'stateful=True' v Keras API omogocimo kljucno znacilnost LSTM - pomnjenje daljsih sekvenc,
# kar ni lastnost tipicne RNN
# LSTM matrika je 3D: [vzorci, koraki, znacilke]
#   vzorci - neodvisna opazovanja, tipicno vrste podatkov
#   koraki - razlicni casovni koraki za dano znacilko za dano opazovanje
#   znacilke - razlicne vrednosti ob casu opazovanja


def lstm_fit(trainSet, batchSize, nbEpoch, neurons):
    # @param batchSize - zaradi manjse racunske zahtevnosti ocenjujemo gradient na vzorcih v velikosti batchSize
    #                  - manjsi kot je, slabsa bo ocena gradienta
    #                  - zaradi paralelizacije je za GPU dober velik batchSize
    # @param nbEpoch - stevilo iteracij ucenja
    #                - 1 epoch je iteracija ucenja cez celotno ucno mnozico (dataset gre naprej in nazaj skozi mrezo)
    #                - pravo stevilo ecpochov omogoca konvergenco modela, preveliko lahko povzroci overfitting
    # @param neurons - stevilo nevronov, za bolj preproste probleme je dovolj 1-5
    # primer: 1000 vzorcev, batchSize=500 -> 1 epoch bo izveden v dveh iteracijah

    X, y = trainSet[:, 0:-1], train[:, -1]      # locimo label
    X = X.reshape(X.shape[0], 1, X.shape[1])    # naredimo 3D np array z eno samo plastjo
    model = Sequential()                        # zgradili bomo sekvencni model

    # batch_input_shape se rabi za stateful model v vhodni plasti
    # ce je plast stateful to pomeni, da bodo stanja, izracunana za trenutni batch zacetna stanja naslednjega batcha
    # to je kljucna lastnost LSTM
    layer = LSTM(neurons, batch_input_shape=(batchSize, X.shape[1], X.shape[2]), stateful=True)
    model.add(layer)                            # sekvencnemu modelu dodamo plast
    model.add(Dense(1))                         # aktivacijsko fjo doloca parameter activation

    # pripravimo model za ucenje
    model.compile(loss='mean_squared_error', optimizer='adam')

    # ucenje v izbranem stevilu iteracij
    for i in range(nbEpoch):

        # shuffle=False ker hocemo da se LSTM uci iz sekvence opazovanj
        model.fit(X, y, epochs=1, batch_size=batchSize, verbose=0, shuffle=False)
        # ker imamo stateful model, moramo rocno na koncu epocha ponastaviti notranje stanje
        model.reset_states()
        print('\repoch {} of {} complete'.format(i+1, nEpoch), end='')

    print("\n")

    # nov model z ustreznim batch_size -> resujem batch_size problem pri forecastu
    nBatch = 1
    forecastModel = Sequential()
    forecastModel.add(LSTM(nNeurons, batch_input_shape=(nBatch, X.shape[1], X.shape[2]), stateful=True))
    forecastModel.add(Dense(1))
    forecastModel.set_weights(model.get_weights())  # nastavimo naucene utezi
    forecastModel.compile(loss='mean_squared_error', optimizer='adam')

    return forecastModel


print("\nFitting the dataset")

# na rezultat ucenja lahko mocno vpliva zacetno stanje modela (dolocal naj bi ga Keras random seed)
modelLSTM = lstm_fit(trainScaled, trainSize, nEpoch, nNeurons)

# VALIDACIJA MODELA
# fixed approach - model naucimo na vseh ucnih podatkih, nato delamo predikcije za vsak element testne mnozice
# dynamic approach - model ponovno naucimo vsakic, ko je na voljo nov podatek iz testne mnozice
# batch_size problem - tensorflow zahteva isto velikost vhodnih podatkov pri predikciji kot je bila velikost batcha
#                      pri ucenju modela. Resitev je, da ustvarimo nov model z utezmi naucenega modela, kjer lahko
#                      predikcije delamo za posamezne vzorce


def lstm_forecast(model, batchSize, X):
    # @param X - pretekli vzorci, na podlagi katerih napovedujemo
    X = X.reshape(1, 1, len(X))
    print(X.shape)
    yPred = model.predict(X, batch_size=1)  # batchSIze ?
    print(type(yPred))
    print(yPred[0, 0])
    return yPred[0, 0]


# validacija modela
print("\nValidation")
yPred = np.empty(len(testScaled))  # hrani rezultate predikcije
yRef = np.empty(len(testScaled))
print(testScaled.shape)

# TODO veckratno ucenje modela za walk-forward validacijo
for i in range(len(testScaled)):

    testX, testY = testScaled[i, 0:-1], testScaled[i, -1]
    testX = testX.reshape(1, 1, len(testX))
    yhat = modelLSTM.predict(testX, batch_size=1)

    invRescaleData = np.concatenate((testX.reshape(1, -1), yhat), axis=1)

    # inverzno skaliranje
    #yhat = inverse_scale_data(MMscaler, testX, yhat)
    yhat = MMscaler.inverse_transform(invRescaleData)

    # shrani napoved
    #print("Sample: {}, Predicted: {}, Expected: {}".format(i+1, yhat[0, -1], test[i, -1]))
    yPred[i] = yhat[0, -1]
    yRef[i] = test[i, -1]

# izracun napake
R2 = r2_score(yRef, yPred)
RMSE = mean_squared_error(yRef, yPred)
print("R^2 result: {}\nRMSE: {}".format(R2, RMSE))

# vizualizacija
plt.figure()
plt.plot(yPred, label='Predicted')
plt.plot(yRef, label='Actual data')
plt.title('Forecast results, epoch number: {}'.format(nEpoch))
plt.legend(loc='best')
plt.show()

print("\nDone")
