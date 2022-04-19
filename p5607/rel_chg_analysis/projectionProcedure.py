import numpy as np

def calcR0(x):
    return np.mean(x)
    

def calcABD(strain, R0, R1, R2):
    # CALCULATE ALPHA, BETA, DELTA NONLINEARITY PARAMETERS
    alpha = 2*R0/((np.max(strain)-np.min(strain))*1e-6)
    beta = 2*R1/((np.max(strain)-np.min(strain))*1e-6)
    delta = 2*R2/((np.max(strain)-np.min(strain))*1e-6)**2  
    return alpha, beta, delta


def orthoTest(Nharm, T, printYN):
    # this loop ensures orthogonality
    for iii in range(2*Nharm):
        if iii > 0:
            for jjj in range(iii-1):
                T[:,iii] = T[:,iii] - np.dot(T[:,jjj],T[:,iii]) * T[:,jjj] # remove cross terms

            T[:,iii] = T[:,iii]/np.sqrt(np.dot(T[:,iii],T[:,iii])) # for all but the first
        else:
            T[:,iii] = T[:,iii]/np.sqrt(np.dot(T[:,iii],T[:,iii])) # for the first one

    # make scalar products to control orthogonality
    orthotest = np.zeros((2*Nharm,2*Nharm))
    for iii in range(2*Nharm):
        for jjj in range(iii, 2*Nharm):
            orthotest[jjj,iii] = np.dot(T[:,iii],T[:,jjj])

    if printYN == 0:
        print('we should get 1 on the diagonal, 0 otherwise\n'+str(orthotest))


def nonlinParams(Nharm, T, dCoverC, dCoverC_strain, wLF, t):
    projection = 0
    projection_strain = 0

    A = np.zeros(2*Nharm); # coefficients
    A_strain = np.zeros(2*Nharm); # coefficients
    # norm f to 1
    # dCoverC=dCoverC/sqrt(dot(dCoverC,dCoverC));
    for jjj in range(len(A)):
        A[jjj] = np.sum(T[:,jjj] * dCoverC); # dCoverC is the signal you want to analyze (dC/C)
        projection = projection + A[jjj]*T[:,jjj];   
        A_strain[jjj] = np.sum(T[:,jjj]*dCoverC_strain) # dCoverC is the signal you want to analyze (dC/C)
        projection_strain = projection_strain + A_strain[jjj]*T[:,jjj] 
    
    R = np.zeros((Nharm))
    R_strain = np.zeros((Nharm))

    for iii in range(Nharm):
        R[iii] = np.sqrt((A[2*iii] * np.max(T[:,2*iii]))**2 + (A[2*iii] * np.max(T[:,2*iii]))**2)
        R_strain[iii] = np.sqrt((A_strain[2*iii]*np.max(T[:,2*iii]))**2 + (A_strain[2*iii]*np.max(T[:,2*iii]))**2)

    signal_projection = 0
    strain_projection = 0

    for iii in range(Nharm):
        signal_projection = signal_projection + A[2*iii]*np.sin(iii*wLF*t)*np.max(T[:,2*iii])+A[2*iii]*np.cos(iii*wLF*t)*np.max(T[:,2*iii])
        strain_projection = strain_projection + A_strain[2*iii]*np.sin(iii*wLF*t)*np.max(T[:,2*iii])+A_strain[2*iii]*np.cos(iii*wLF*t)*np.max(T[:,2*iii])

    return projection_strain, R, R_strain, signal_projection, strain_projection


def projectionProc(t, V_HF, strain, Nharm, fn):
    R0 = calcR0(V_HF)
    dCoverC = V_HF - R0
    R0_strain = calcR0(strain)
    dCoverC_strain = strain - R0_strain

    # matrix containing the basis of orthogonal functions
    T = np.zeros((len(t),2*Nharm)) 

    # Freq will be the frequency of the pump
    wLF = 2 * np.pi * fn 

    for iii in range(Nharm):
        T[:,2*iii] = np.sin((iii+1)*wLF*t) # rows of sines and cosines
        T[:,2*iii+1] = np.cos((iii+1)*wLF*t)

    orthoTest(Nharm, T, 0)

    # CALCULATE NONLINEARITY PARAMETERS
    projection_strain, R, R_strain, signal_projection, strain_projection = nonlinParams(Nharm, T, dCoverC, dCoverC_strain, wLF, t)
    R1 = R[0]
    R2 = R[1]
    alpha, beta, delta = calcABD(projection_strain, R0, R1, R1)

    return strain_projection,signal_projection,alpha,beta,delta,R0,R1,R2