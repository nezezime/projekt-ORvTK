import matplotlib.pyplot as plt
import numpy as np
import pickle
from ts_regress import ts_regress, ts_regress_eval

# TODO list
# normalizacija casovnih vrst (naj bodo cim bolj stacionarne) - kot v laboratorijski vaji
# vizualizacija podatkov

# variables
uIDs = [10, 19, 20, 27, 34]  # IDji uporabnikov
userCount = len(uIDs)

# read pickle python format to dictionaries
# pickle - serializes a python object into stream of bytes
# vsak dictionary ima pet polj - za vsakega uporabnika eno
reg_arousal = pickle.load(open('arousal.pckl', "rb"))               # vzburjenost/vznemirjenje
reg_valence = pickle.load(open('valence.pckl', "rb"))               # valenca/prisotnost
reg_times = pickle.load(open('times.pckl', "rb"))                   # casovne znamke
reg_emo_mouth = pickle.load(open('mouth.pckl', "rb"))               # usta
reg_leftPupilDiameter = pickle.load(open('Lpupil.pckl', "rb"))      # premer leve zenice
reg_rightPupilDiameter = pickle.load(open('Rpupil.pckl', "rb"))     # premer desne zenice

# dictionary se bere po kljucu
# vsak uporabnik ima razlicno dolzino casovne vrste
uDataSize = [len(reg_arousal[10]), len(reg_arousal[19]),
             len(reg_arousal[20]), len(reg_arousal[27]),
             len(reg_arousal[34])]
maxDataSize = 3924
print(uDataSize)

# NASTAVLJANJE MODELA
# dolocimo, koliko vzorcev nazaj upostevamo
nBack = 10

# dolocimo velikost podatkovnega seta
dataSetLen = 1000

# dolocimo, katero znacilko iscemo (label)
Y = reg_arousal

# dolocimo vhodne znacilke
# sestavimo podatkovni set
X = np.zeros(shape=[len(uIDs), dataSetLen, 3])
for idx, val in enumerate(uIDs):
    X[idx] = np.array([reg_emo_mouth[val][0:dataSetLen],
                      reg_leftPupilDiameter[val][0:dataSetLen],
                      reg_rightPupilDiameter[val][0:dataSetLen]
                       ]).T

print("data shape: \n")
print(X.shape)

R2 = np.empty(shape=[userCount, nBack, nBack, nBack])  # hrani R^2 za vse kombinacije n
R2SeriesAllUsers = np.empty(shape=[userCount, nBack * nBack * nBack])

# izvedemo ucenje za vsakega uporabnika
# regresija z metodo najmanjsih kvadratov (statsmodels.api.OLS)
for idx, uID in enumerate(uIDs):
    print "computation running for user", uID

    R2Series = np.empty(nBack * nBack * nBack)

    # regresijo izvajamo za vse mozne kombinacije propagacij
    cnt = 0
    for n1 in range(0, nBack):
        for n2 in range(0, nBack):
            for n3 in range(0, nBack):
                y = Y[uID][0:dataSetLen]                                              # nastavimo dolzino label
                p_lags = [n1 + 1, n2 + 1, n3 + 1]
                mod_ft = ts_regress(y, X[idx], p_lags)     # metoda najmanjsih kvadratov, tip: statsmodels.regression.linear_model.RegressionResultsWrapper
                R2[idx][n1][n2][n3] = mod_ft.rsquared_adj
                R2Series[cnt] = mod_ft.rsquared_adj
                cnt += 1

    R2SeriesAllUsers[idx] = R2Series

# dolocimo uspesnost prileganja modela - R^2
# pove, koliko bolje se nas model prilega glede na povprecenje vrednosti (konstantni model)

# PRIKAZ REZULTATOV
print "\nRESULTS"
print(R2SeriesAllUsers.min())
print(R2SeriesAllUsers.max())
#print(R2[0])

# maksimalni R^2 za vsakega uporabnika








