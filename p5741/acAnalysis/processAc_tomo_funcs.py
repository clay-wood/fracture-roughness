import numpy as np
import scipy.io as sio
import h5py as h5
import pandas as pd
import psutil
import os
import fnmatch

from bokeh.plotting import figure, show
from bokeh.io import output_notebook
from bokeh.layouts import gridplot, row, column
output_notebook()


#----------------------------------------------------------------------
# FILE I/O
#----------------------------------------------------------------------

def read_hdf(fileName):
    """
    INPUT
    fileName: path to file hdf5 file
    
    OUTPUT
    iterate through keys in hdf5 and insert into pandas dataframe
    """
    data = pd.DataFrame()
    for chanName in h5.File(fileName, "r").keys():
        data[chanName] = h5.File(fileName, "r")[chanName]
    return data


def loadACsettings(AcSettingsfile):
    """
    INPUT
    AcSettingsfile: path to Acoustic Settings File from VSX
    
    OUTPUT
    Load pertinent acoustic settings from .mat file return as array 
    """
    acSettings = sio.loadmat(AcSettingsfile) # load acoustic settings
    numSFpfile = acSettings['numFrames'][0,0]/2 # number of superframes per file
    numWFpSFpCH = acSettings['numAcqs'][0,0] # number of WF per superframe and per channel
    numWFpfilepCH = numSFpfile*numWFpSFpCH; # number of WF per file and per channel
    numCHR = np.size(acSettings['channels2save'][0]) # number of channels
    numCHT = np.size(acSettings['channels2transmit'][0]) # number of channels
    WFlength = acSettings['Nsamples'][0,0] # waveform length
    sampFreq = acSettings['samplingFreq'][0,0] # waveform length
    del acSettings
    return numSFpfile, numWFpSFpCH, numWFpfilepCH, numCHR, numCHT, WFlength, sampFreq


def LoadAcFile(WF_path,filenumber,numCHR,numSFpfile):

#     Load acoustic file and reshape according to the
#     number of receivers
#     WF_path: location of the files
#     filenumber: which file is being loaded
#     numCHR: number of receivers 
#     numSFpfile: number of 'superframes' per file (Verasonics jargon)

    with open(WF_path+np.str(int(filenumber))+'.ac', 'rb') as fid:
            
        ACdata = np.array(np.fromfile(fid, np.int16)) 
        
        # reshape to get one column per channel
        ACdata = np.reshape(ACdata,(int(numSFpfile),numCHR,-1)) # 3D matrix with WF vs numCHR vs number of SF
        ACdata = np.rollaxis(ACdata,1,0) # put numCHR as the last dimension before reshaping
        ACdata = ACdata.reshape((numCHR,-1)) # WF vs numCHRs
        
    fid.close();
    
    return ACdata.T


def folder_size(path):
    """
    INPUT
    fileName: path to file hdf5 file
    
    OUTPUT
    iterate through keys in hdf5 and insert into pandas dataframe
    """
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += folder_size(entry.path)
            
    numAcFiles = len(fnmatch.filter(os.listdir(path), '*.ac'))
    return numAcFiles, np.round(total/1e9, 2)


def calcWFchunks(WF_path):
    numAcFiles = len(fnmatch.filter(os.listdir(WF_path[0:-3]), '*.ac'))
    dir_size = folder_size(WF_path[0:-3])
    # int(dir_size / numAcFiles)
    WF1_size = os.path.getsize(WF_path[0:-3]+'WF_1.ac')
    print('There are '+np.str(numAcFiles)+' files in ' + str(WF_path[0:-3]))
    print('Size of Directory: '+str(np.round(dir_size/1e9,2))+' GB')

    avail_mem = psutil.virtual_memory().available
    WF_chunk = int(np.floor((0.75*avail_mem)/WF1_size))
    print('Available Memory: '+str(np.round(avail_mem / 1e9,2))+' GB')

    if dir_size > avail_mem:
        print('\nThere will be '+str(WF_chunk)+' *.ac files read into memory at a time.')
    else: 
        print('\nThe entire directory of *.ac files will be read into memory.')
    return numAcFiles, WF1_size, WF_chunk


def saveData(runname, acousticrun, LocalAcTime, NtoStack, ref, TimeRange, datakeys, acdata):
    if len(TimeRange) == 0:
        filenamedata = "Results_"+runname+"_"+acousticrun+"_"+str(LocalAcTime[0,0])+"s-"+str(LocalAcTime[-1,-1])+"s_Stack"+str(NtoStack)+"WFs_"+ref[0]+"w_Amp.hdf5"
    else:
        filenamedata = "Results_"+runname+"_"+acousticrun+"_fullrun_Stack"+str(NtoStack)+"WFs_"+ref[0]+"w_Amp.hdf5"

    with h5.File(filenamedata, "w") as f:
        for aa, chanName in enumerate(datakeys):
            f.create_dataset(chanName, data=acdata[aa])

#----------------------------------------------------------------------
# FILE I/O
#----------------------------------------------------------------------

def findidxs(acTime,Time,TransNum,numCHT,numWFpfilepCH):

    """Use this function to find indexes corresponding to a certain Time within the run and corresponding to transmitter "TransNum".

    Inputs:
    acTime: Sync Time vector, output of SyncAcData.
    Time: time at which we want the indexes (the closest indexes to that time will be found).
    TransNum: The indexes provided corresponds to a waveform recorded when
    Transmitter TransNum was active.
    numCHT: number of transmitters used during the run
    numWFpfilepCH: number of WF per receiver and per file

    Outputs:
    filenumber corresponding to the provided Time
    idxWFwithinfile: idx of the WF of interest within filenumber
    idxAcTime: idx of the WF within the acTime vector
    ExactTime: exact time corresponding to the index found"""

    [mini,idxAcTime] = [np.amin(np.abs(acTime-Time)), np.argmin(np.abs(acTime-Time))] # find nearest idx within acTime

    ShiftTrans = np.mod(idxAcTime,numCHT);
    if ShiftTrans == 0:
        ShiftTrans = numCHT;  # i.e. no shift
    
    idxAcTime = idxAcTime + (TransNum - ShiftTrans); # closest WF within that file corresponding to transmitter TransNum

    ExactTime = acTime[idxAcTime];

    filenumber = np.ceil(idxAcTime/numWFpfilepCH); # corresponding file number
    idxWFwithinfile = np.mod(idxAcTime,numWFpfilepCH); # closest WF within the first file

    return [filenumber,idxWFwithinfile,idxAcTime,ExactTime]


def rms(X, ax):
    """
    INPUT
    X: vector 
    ax: axis to calculate root mean square
    
    OUTPUT
    return rms along given axis
    """
    return np.sqrt(np.nanmean(X**2,axis=ax))


def amp(X, ax):
    """
    INPUT
    X: vector 
    ax: axis to calculate maximum, relative to minimum
    
    OUTPUT
    return peak-to-peak of vector along given axis
    """
    return (np.amax(X, axis=ax)-np.amin(X, axis=ax))



def ShowMeWFs(WF_path,AcSettingsfile,SyncFile,AtWhichTimes,NtoStack,Offset,WhichTransTomoMode,IdxWindow):

    # This function is used to display waveforms at particular times during the run, 
    # with the options of stacking them. It is typically used to pickarrival times by hand, 
    # or decide how many waveforms need to be stacked before further processing

    # acoustic parameters
    numSFpfile, numWFpSFpCH, numWFpfilepCH, numCHR, numCHT, WFlength, sampFreq = loadACsettings(AcSettingsfile)



    # MAYBE THIS DOES NOT NEED TO BE IMPLEMENTED AT THE MOMENT
    # if ~isequal(size(IdxWindow),[2 numCHR numCHT]) && ~isequal(size(IdxWindow),[0 0]) && ~isequal(size(IdxWindow),[1 2]) && ~isequal(size(IdxWindow),[2 1])
    #     display(['Size of IdxWindow is not correct. Size should either be [2 by ' num2str(numCHR) ' by ' num2str(numCHT) '], or empty, or a two element vector.']);

    # # if only a two element vector is provided, use it for all combinations of TR
    # if isequal(size(IdxWindow),[1 2]) || isequal(size(IdxWindow),[2 1])
    #     IdxWindowInit = IdxWindow;
    #     IdxWindow = repmat(IdxWindowInit,1,numCHR,numCHT);
    # end


    # Load sync data
    # [acTime, acPeriod_adjusted, ts, TotalNumberOfFiles] = list(map(lambda x: np.load(SyncFile)[x], ['arr_0','arr_1','arr_2','arr_3']))
    syncdata = read_hdf(SyncFile)
    fs = 1/syncdata["ts_adjusted"].values # acoustic sampling rate
    
    if WhichTransTomoMode > numCHT:    
        print('You chose to display transmitter #'+str(WhichTransTomoMode)+'. Please choose a transmitter between 1 and '+str(numCHT)+'.');
    # IT WOULD BE NICE TO RETURN AN ERROR MESSAGE HERE    

    # time vector for each waveform
    timeWF = np.arange(0,WFlength)*syncdata["ts_adjusted"].values[0]

    # number of WFs to show
    N = np.size(AtWhichTimes)
    rangeTimes = np.zeros((N,2))

    p = figure(title='WFs to Stack', tools='pan,box_zoom,undo,save,hover', plot_width=1200, plot_height=800) # initialize plot for WFs

    for ii in range(0,N):
        idxAcTime = np.where(syncdata["acTime"].values > AtWhichTimes[ii])[0][0];      
        idxAcTimeVec = np.arange(idxAcTime-np.ceil(NtoStack/2)+1, idxAcTime+np.floor(NtoStack/2), dtype=int); # vector of indexes centered around 'idxAcTime'    
        if idxAcTimeVec[0] < 1:
            print('Either the first value of ''AtWhichTimes'' is too small or ''NtoStack'' is too large. Can not stack '+np.str(NtoStack)+' waveforms centered around '+np.str(AtWhichTimes[0])+ ' s.');
        elif idxAcTimeVec[-1] > np.size(syncdata["acTime"].values):
            print('Either the last value of ''AtWhichTimes'' is too large or ''NtoStack'' is too large. Can not stack '+np.str(NtoStack)+' waveforms centered around '+np.str(AtWhichTimes[-1])+' s.');

        rangeTimes[ii,:] = np.array([syncdata["acTime"].values[idxAcTimeVec[0]], syncdata["acTime"].values[idxAcTimeVec[-1]]]);     

        [filenumber,idxWFwithinfile,idxExactTime,ExactTime] = findidxs(syncdata["acTime"].values,rangeTimes[ii,1],WhichTransTomoMode,numCHT,numWFpfilepCH);

        fullWFref = np.zeros([WFlength,numCHR]);
        for kk in range(0,NtoStack): # number of WFs to stack                                

            if np.logical_or(idxWFwithinfile <= numCHT, kk == 0): # open new file if idxWFwithinfile is 1 or if it's the first WF to stack
                ACdata = LoadAcFile(WF_path,int(filenumber),numCHR,numSFpfile);

            fullWFref = fullWFref + ACdata[int(WFlength*(idxWFwithinfile-1)):int(WFlength*idxWFwithinfile),:]; # stack WFs

            if idxWFwithinfile <= numWFpfilepCH - numCHT: # stay within the same file for the next loop
                idxWFwithinfile = idxWFwithinfile + numCHT; # update to the next WF corresponding to the same transmitter 
            else:                                # use next file for the next loop
                filenumber = filenumber + 1; # go to next file
                idxWFwithinfile = WhichTransTomoMode; # start in the next file at WFs corresponding to transmitter "WhichTransTomoMode"

        fullWFref = fullWFref/NtoStack;    

        colors = np.array(['red','orange','blue'])
        if N <= 3:
            for chnum in range(numCHR):
                p.line(timeWF, fullWFref[:,chnum]-Offset*(chnum), line_color=colors[ii], line_width = 3, legend_label='WF_'+str(ii+1))
        else:
            for chnum in range(numCHR):
                p.line(timeWF, fullWFref[:,chnum]-Offset*(chnum), line_width = 3, legend_label='WF_'+str(ii+1))
    p.yaxis.axis_label = 'Amplitude'
    p.xaxis.axis_label = 'Time (us)'        
    p.legend.click_policy='hide'
    show(p)


# def getfullWFref2(NtoStack, numWFpSFpCH, WF_path, numSFpfile, WFlength, numCHR, numCHT):

#     StackToNumWFs = NtoStack/(numWFpSFpCH*2)
#     if StackToNumWFs <= 1:
#         WFdat = LoadAcFile(WF_path,1,numCHR,numSFpfile)
#         WFdat = np.reshape(WFdat,(numCHT,-1,WFlength,numCHR))
#         WFdat = np.moveaxis(WFdat,-1,0)
#         WFdat = np.moveaxis(WFdat,-1,0)
#         fullWFref = np.sum(WFdat[:,:,:,0:NtoStack],axis = -1)
#     else: 
#         numRefFiles = np.ceil(StackToNumWFs).astype(int)
#         WFdat = np.empty((numRefFiles,(numWFpSFpCH*2*8*WFlength),numCHR))
#         for aa in range(numRefFiles):
#             WFdat[aa,:,:] = LoadAcFile(WF_path,aa,numCHR,numSFpfile)
#         WFdat = np.reshape(WFdat,(numCHT,-1,WFlength,numCHR))
#         WFdat = np.reshape(WFdat,(numRefFiles,numCHT,-1,WFlength,numCHR))
#         WFdat = np.moveaxis(WFdat,-1,2)
#         WFdat = np.moveaxis(WFdat,3,4)
#         WFdat = np.reshape(WFdat,(numCHT,numCHR,WFlength,-1))
#         fullWFref = np.sum(WFdat[:,:,:,0:NtoStack],axis = -1)

#     return fullWFref/NtoStack/reference_NrefWF


def getfullWFref(kk, jj, ii, upperlimit, filenumber1, idxWFwithinfile1, WFlength, WF_path, numCHR, numCHT, chnumt, numSFpfile, numWFpfilepCH, NtoStack, reference_NrefWF):
    fullWFref = np.zeros((WFlength,numCHR,numCHT))
    while kk < upperlimit:
        if np.logical_or((jj == 0), np.logical_and(ii == filenumber1, jj == idxWFwithinfile1)): # open new file if jj = 1 or if it's the first file
            ACdata = LoadAcFile(WF_path,ii,numCHR,numSFpfile)       
 
        fullWFref[:,:,chnumt] = fullWFref[:,:,chnumt] + ACdata[WFlength*(jj-0):WFlength*(jj+1),:] # read data
    
        if chnumt < (numCHT-1): # chnumt runs from 1 to numCHT
            chnumt = chnumt + 1
        else:
            chnumt = 0

        if jj < (numWFpfilepCH-1):  # stay within the same file for the next run
            jj = jj + 1
        else:                   # use next file for the next run
            jj = 0
            ii = ii + 1
        kk = kk + 1
    
    return fullWFref/NtoStack/reference_NrefWF




def xcorr(a, b):
    norm_a = np.linalg.norm(a)
    a = a / norm_a
    norm_b = np.linalg.norm(b)
    b = b / norm_b
    return np.correlate(a, b, mode = 'full')



def delay(X, ts, plot01):

    NN = (len(X)+1)/2
    tps = np.arange(-NN*ts,(NN+1)*ts,ts)
    [maxi,ind] = [np.amax(X),np.argmax(X)]
    [x1, x2, x3] = [ind-1, ind, ind+1]
    
    if x1 == 0:
        x1 = 1
        x2 = 2
        x3 = 3
    elif x3 == len(X):
        x1 = len(X) - 3 # arbitrarily choose the maximum at x2 = length(X) - 1;
        x2 = len(X) - 2
        x3 = len(X) - 1
        ind = len(X) - 1; 
        
    [y1 ,y2, y3] = X[[x1, x2, x3]]
    b = ((y1 - y3)*(x1**2 - x2**2) - (y1 - y2)*(x1**2 - x3**2))/((x1 - x3)*(x1**2 - x2**2) - (x1 - x2)*(x1**2 - x3**2))
    a = (y1 - y3 - b*(x1 - x3))/(x1**2 - x3**2)
    c = y1 - a*x1**2 - b*x1

    ind_max_corr = -b/(2*a)
    max_interpolation = a*ind_max_corr**2 +b*ind_max_corr + c

    timedelay = ts*(ind_max_corr - NN)
    interpX = np.arange(x1, x3+1, 0.01)
    interpY = a*interpX**2 + b*interpX + c
    interpT = ts*(interpX - NN+1)

    if plot01 == 1:
        fig1 = figure(title='WFs to Stack', tools='pan,box_zoom,undo,save,hover', plot_width=1200, plot_height=800) 
        fig1.circle(tps[ind], maxi, size = 10, fill_color='red', line_width = 0, legend_label='max of intercorr') 
        fig1.circle(timedelay, max_interpolation, size = 10, fill_color='limegreen', line_width = 0, legend_label='refined max of intercorr')
        fig1.circle(tps[NN], X[NN], size = 10, fill_color='blue', line_width = 0, legend_label='interpolation')
        fig1.circle(interpT, interpY, size = 10, fill_color='black', line_width = 0)
        fig1.yaxis.axis_label = 'Time between consecutive triggers (ms)'
        fig1.xaxis.axis_label = 'Index Number'
        show(fig1)
    
    return [max_interpolation, timedelay]


def interpDat(Time, LocalAcTime, data, idxAc, numCHR, numCHT):
    dataI = np.zeros((len(Time[idxAc]),numCHR,numCHT))

    # Interpolate
    for chnumr in range(numCHR):
        for chnumt in range(numCHT):
            dataI[:,chnumr, chnumt] = np.interp(Time[idxAc], LocalAcTime[:,chnumt], data[:,chnumr,chnumt])

    return dataI


def processAc(WF_path,AcSettingsfile,AcSyncFile,idx2analyze,ref,NtoStack,threshold,Offset2,displayoptions,Filter,TimeRange,NZPad,FreqQ):

    # from findidxs import *
    from scipy.signal import firwin
    # from processAc_tomo_funcs import getfullWFref, rms, amp, calcWFchunks

    reference_type = ref[0]
    reference_NrefWF = ref[1]
    # SyncFile = AcSyncFile
    IdxWindow = idx2analyze

    if reference_type == 'absref':
        threshold = -1
        print('The threshold is set to -1 (i.e., no threshold) when using ''absref''.')  

    numSFpfile, numWFpSFpCH, numWFpfilepCH, numCHR, numCHT, WFlength, sampFreq = loadACsettings(AcSettingsfile)
    # [acTime, acPeriod_adjusted, ts, TotalNumberOfFiles] = list(map(lambda x: np.load(SyncFile)[x], ['arr_0','arr_1','arr_2','arr_3']))
    syncdata = read_hdf(AcSyncFile)

    # --------------------------------------------------------------------------------------

    # find the files and WFs within the files where analysis begins and ends
    if len(TimeRange) == 0:    
        filenumber1 = 1 # first file
        filenumber2 = (syncdata["TotalNumberOfFiles"].values[0]) # last file
        idxWFwithinfile1 = 1 # first WF to be analyzed within the first file
        idxWFwithinfile2 = numWFpfilepCH # last WF to be analyzed within the last file
        idxacTime1 = 1
        idxacTime2 = len(syncdata["acTime"].values)
    else:    
        [filenumber1,idxWFwithinfile1,idxacTime1,acTime1] = findidxs(syncdata["acTime"].values,int(TimeRange[0]),1,numCHT,numWFpfilepCH)
        [filenumber2,idxWFwithinfile2,idxacTime2,acTime2] = findidxs(syncdata["acTime"].values,int(TimeRange[1]),numCHT,numCHT,numWFpfilepCH)     
        idxWFwithinfile1 = int(idxWFwithinfile1)
        idxWFwithinfile2 = int(idxWFwithinfile2)

    print('From WF '+str(idxWFwithinfile1)+' of file '+str(filenumber1)+' to WF '+str(int(idxWFwithinfile2))+' of file '+str(int(filenumber2))+'.')

    # --------------------------------------------------------------------------------------

    acN = idxacTime2 - idxacTime1 + 1 # total number of WF per receiver
    acN = int(np.floor(acN/NtoStack/numCHT)) # total number of stacked WF per receiver and per transmitter. "floor" because NtoStack is not necessarily a multiple of the total number of waveforms

    # define LocalAcTime to account for stacking, the number of transmitters
    # used and the time range where the analysis is conducted

    acPeriod_new = NtoStack*syncdata["acPeriod_adjusted"].values[0]*numCHT
    LocalAcTime = np.arange(0,acN)*acPeriod_new/1e6 # acPeriod is in microsec

    acTime_newshifted = np.zeros((len(LocalAcTime),numCHT)) # matrix containing one time vector per transmitter
    for chnumt in range(0,numCHT): # shift by the average time of all stacked WFs
        acTime_newshifted[:,chnumt] = LocalAcTime + np.mean([syncdata["acTime"].values[idxacTime1+chnumt-1], syncdata["acTime"].values[idxacTime1+chnumt-1+(NtoStack-1)*numCHT]])          

    LocalAcTime = acTime_newshifted

    # time vector for each waveform
    timeWF = np.arange(0,WFlength)*syncdata["ts_adjusted"].values[0]

    # fullWFref = np.zeros((WFlength,numCHR,numCHT))
    RmsAmpRef = np.zeros((numCHR,numCHT))
    AmpRef = np.zeros((numCHR,numCHT))

    # filter to be used if noisy waveforms (adjust order and frequencies as needed)
    #NEED TO CHECK IF THE FILTER WORKS PROPERLY
    # if Filter == 1:
    #     filterparam = firwin(Filter_order, Filter_frq*syncdata["ts_adjusted"].values[0]*2, pass_zero='lowpass') # (ts is in microsec)

        
    # build a reference waveform

    # if 'absref' is chosen, all waveforms in the time range are stacked
    # to build a template

    # if 'relref' or 'mixref' is chosen, this reference WF will be used only once to
    # be compared with the next one

    kk = 0 # from 0 to NtoStack*numCHT - 1 for relref and mixref or from 0 to acN*numCHT - 1 fr absref 
    ii = filenumber1 # file number
    jj = idxWFwithinfile1 # from 1 to numWFpfilepCH 
    chnumt = 0 # transmitter index

    # if (strcmp(reference_type,'relref')||strcmp(reference_type,'mixref')) && reference_NrefWF == -1
    #     print('reference.NrefWF cannot be -1 with relative or mixed reference');
        
    if (reference_type == 'relref' or reference_type == 'mixref') and reference_NrefWF == -1:
        print('reference.NrefWF cannot be -1 with relative or mixed reference')

    if reference_type == 'absref' and reference_NrefWF == -1:
        reference_NrefWF = acN

    upperlimit = NtoStack*numCHT*reference_NrefWF # number of WF used to built a reference   

    fullWFref = getfullWFref(kk, jj, ii, upperlimit, filenumber1, idxWFwithinfile1, WFlength, WF_path, numCHR, numCHT, chnumt, numSFpfile, numWFpfilepCH, NtoStack, reference_NrefWF)

    # --------------------------------------------------------------------------------------

    # IMPLEMENT FILTERING IN PYTHON
    # if Filter == 'yes':
    #     fullWFrefF = filtfilt(filterparam,1,fullWFref); # filtering
    #     if Filter_view == 0:
    #         figure(765);plot(fullWFref(:,6,5));hold on; # plot pair R1-T1 unfiltered
    #         plot(fullWFrefF(:,6,5),'k');hold off; # plot pair R1-T1 filtered
            
    #     fullWFref = fullWFrefF;
    # -----------------------------------------------------------------------------

    # if only a two element vector is provided, use it for all combinations of TR
    if IdxWindow.ndim < 2:
        IdxWindowInit = IdxWindow
        IdxWindow = np.tile(IdxWindowInit,(numCHR,numCHT))

    # isolate windows to be analyzed
    windows_length = IdxWindow[:,:,1] - IdxWindow[:,:,0]
    maxwindowlength = np.max(windows_length)

    WFref = np.zeros((maxwindowlength,numCHR,numCHT))

    for chnumt in range(0,numCHT):
        for chnumr in range(0,numCHR):        
            WFref[0:windows_length[chnumt,chnumr],chnumr,chnumt] = fullWFref[IdxWindow[chnumt,chnumr,0]:IdxWindow[chnumt,chnumr,1],chnumr,chnumt] # part of the WF to be analyzed

    for chnumr in range(0,numCHR):
        RmsAmpRef[chnumr,:] = rms(WFref[:,chnumr,:],0) # RmsAmp of the reference waveform
        AmpRef[chnumr,:] = amp(WFref[:,chnumr,:],0) # Peak-to-Peak Amp of the reference waveform

    # del ACdata
    # return [WFref, windows_length, maxwindowlength, RmsAmpRef, AmpRef]

    # --------------------------------------------------------------------------------------

    # keep initial ref WFs in memory to save it at the end of the function
    # (necessary in case they are changed when using relref or mixref)
    firstfullWFref = fullWFref; 
    # return # uncomment here to look at the reference waveform

    ## Compute changes in time of flight, max of intercorrelation and RmsAmp

    MaxInter = np.zeros((acN,numCHR,numCHT))
    TimeShift = np.zeros((acN,numCHR,numCHT))
    RmsAmp = np.zeros((acN,numCHR,numCHT))
    Amp = np.zeros((acN,numCHR,numCHT))
    freqQAmp = np.zeros((acN,numCHR,numCHT))
    maxAmp = np.zeros((acN,numCHR,numCHT))
    maxFreq = np.zeros((acN,numCHR,numCHT))

    ii = filenumber1 # file number
    jj = idxWFwithinfile1 # from 1 to numWFpfilepCH

    # adjust depending on what you want to display
    # h1 = np.zeros((numCHR,numCHT))
    # h2 = np.zeros((numCHR,numCHT))
    # h3 = np.zeros((numCHR,numCHT))
    # h4 = np.zeros((numCHR,numCHT))

    # FigXcorr = figure;
    # set(gca,'YTickLabel',[]);
    # set(gca,'Ylim',[-numCHT*numCHR*Offset Offset]);
    # set(gca,'NextPlot','replacechildren');

    # --------------------------------------------------------------------------------------

    for hh in range(0, acN): # from 1 to the total number of stacked waveforms
        fullWF = np.zeros((WFlength,numCHR,numCHT));   
        """stack WFs"""

        chnumt = 0 # transmitter index
        for kk in range(0, (NtoStack*numCHT)):
            if np.logical_or((jj == 0), np.logical_and(ii == filenumber1, jj == idxWFwithinfile1)): # open new file if jj = 1 or if it's the first file
            # if (jj == 1) || (ii == filenumber1 && jj == idxWFwithinfile1): # open new file if jj = 1 or if it's the first file
                # print('File number '+str(ii)+'.\n') # display file number
                # print(jj)
                ACdata = LoadAcFile(WF_path,ii,numCHR,numSFpfile)
                
            fullWF[:,:,chnumt] = fullWF[:,:,chnumt] + ACdata[WFlength*(jj-0):WFlength*(jj+1),:] # read data
            if chnumt < (numCHT-1): # chnumt runs from 1 to numCHT
                chnumt = chnumt + 1
            else:
                chnumt = 0
        
            
            if jj < (numWFpfilepCH-1):   # stay within the same file for the next run
                jj = jj + 1
            else:                    # use next file for the next run
                jj = 0
                ii = ii + 1
                
        fullWF = fullWF/NtoStack #3 stacked WF
        
        """WF is the part to be analyzed"""
        WF = np.zeros((maxwindowlength,numCHR,numCHT))
        for chnumt in range(0, numCHT):
            for chnumr in range(0, numCHR):
                WF[0:windows_length[chnumr,chnumt],chnumr,chnumt] = fullWF[IdxWindow[chnumr,chnumt,0]:IdxWindow[chnumr,chnumt,1],chnumr,chnumt] # part of the WF to be analyzed 
        
        """cross-correlate (time delay)"""
        for chnumt in range(0, numCHT):
            for chnumr in range(0, numCHR):     
                corr_signals = xcorr(WFref[:,chnumr,chnumt],WF[:,chnumr,chnumt])
                MaxInter[hh,chnumr,chnumt],TimeShift[hh,chnumr,chnumt] = delay(corr_signals,syncdata["ts_adjusted"].values[0],0)
        
        """amplitudes (time domain)"""
        RmsAmp[hh,:,:] = rms(WF,0)              # RmsAmp of the waveform
        Amp[hh,:,:] = amp(WF, 0) # Max Amp of the waveform
        
        
        # """amplitudes (frequency domain)"""
        # [freqQAmp(hh,:,:), maxAmp(hh,:,:), maxFreq(hh,:,:)] = AmplitudeFreq(sampFreq, WF, NZpad,freqQ);
    
        if (hh/1000 == round(hh/1000)): # display Max intercorrelation every 100 stacked waveforms 
        # if (hh/1000 == round(hh/1000)) & (printMaxInter == 1): # display Max intercorrelation every 100 stacked waveforms
            print(np.squeeze(MaxInter[hh,:,:]))

    """take the opposite such that a positive TimeShift means later arrival."""
    TimeShift = -1*TimeShift     


    return MaxInter, TimeShift, RmsAmp, Amp, RmsAmpRef,AmpRef, fullWFref, LocalAcTime, freqQAmp, maxAmp, maxFreq