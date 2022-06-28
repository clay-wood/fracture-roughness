import numpy as np
import pandas as pd
from scipy.signal import lfilter, butter, filtfilt
from sklearn.metrics import r2_score
import h5py as h5

#---------------------------------------------------------------------------------------
# IMPORT, EXPORT DATA IN HDF5 FORMAT

def readHDF(fileName):
    data = pd.DataFrame()
    for chanName in h5.File(fileName, "r")['mechData'].keys():
        data[chanName] = h5.File(fileName, "r")['mechData'][chanName]
    return data


def saveHDF(filename, data):
    with h5.File(filename+".hdf5", "w") as f:
        for chanName in data.keys():
            f.create_dataset(chanName, data=data[chanName])

#---------------------------------------------------------------------------------------

def detrend(x, y):
    '''Simple linear fit and detrend'''
    fin_idx = np.isfinite(y)
    p = np.polyfit(x[fin_idx], y[fin_idx], 1)
    q = np.polyval(p, x)
    y_detrend = (y - q) + y[fin_idx][0]
    return y_detrend


def butter_filter(data, cutoff, fs, order, hilo):
#     print("Cutoff freq " + str(cutoff))
    nyq = 0.5 * fs # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    '''Get the filter coefficients''' 
    b, a = butter(order, normal_cutoff, btype=hilo, analog=False)
    y = filtfilt(b, a,data)
    return y


def movingmean(x, N):
    # cumsum = np.cumsum(np.insert(x, 0, 0)) 
    # return (cumsum[N:] - cumsum[:-N]) / float(N)
    mm = np.convolve(x, np.ones((N,))/N, mode='same')
    return mm


def movingslope(vec,supportLength,modeLorder = 1,dt = 1):    
    '''Implementation of Moore-Penrose pseudo-inverse method. Adapted from MATLAB Central movingslope package.'''

    def getcoef(t,supportLength,modeLorder):
        A = np.tile(t,(2,modeLorder)).transpose()
        B = np.tile(range(0,modeLorder + 1),(supportLength,1))
        return np.linalg.pinv(np.power(A,B))[1,]

    n = len(vec)
#     if((supportLength <=1) or (supportLength > n) or(supportLength != np.floor(supportLength))):
#         print "supportlength must be a scalar integer, >= 2, and no more than length(vec)"
#     if((modeLorder < 1) or (modeLorder > min(10,supportLength - 1)) or (modeLorder != np.floor(modeLorder))):
#         print "modelorder must be a scalar integer, >= 1, and no more than min(10,supportlength - 1)"
#     if(dt < 0):
#         print "dt must be a positive scalar numeric variable"

    if (supportLength % 2 == 1):
        parity = 1
    else:
        parity = 0

    s = int((supportLength - parity)/2)
    t = np.arange((-s + 1 - parity),s + 1)
    coef = getcoef(t,supportLength,modeLorder)

    f = lfilter(-coef,1,vec)

    Dvec = np.zeros(len(vec))
    Dvec[s:s + n - supportLength + 1] = f[supportLength - 1::]

    for i in range(0,s):
        t = np.arange(0,supportLength)
        t[:] = [x - i for x in t]
        coef = getcoef(t,supportLength,modeLorder)
        Dvec[i] = np.dot(coef,vec[0:supportLength])        

        if(i < s + parity - 1):
            t = np.arange(0,supportLength)
            t[:] = [x - supportLength + i for x in t]
            coef = getcoef(t,supportLength,modeLorder)
            Dvec[n - i - 1] = np.dot(coef,vec[n - supportLength:n])

    Dvec = Dvec/dt
    return Dvec


def getStrain(vert_dcdt, lay_thck):
    '''Engineering Shear Strain: (Hecky Strain)'''
    shrStrain = np.nancumsum((vert_dcdt.to_numpy()[1::] - vert_dcdt.to_numpy()[0:-1]) / ((lay_thck.to_numpy()[1::]+lay_thck.to_numpy()[0:-1])/2.0))
    shrStrain = np.insert(shrStrain, 0, 0)
    return shrStrain


