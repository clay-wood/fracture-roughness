import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

def sample():
    fig, ax = plt.subplots()
    ax.set_title('click on points')

    line, = ax.plot(np.random.rand(100), 'o',
                    picker=True, pickradius=5)  # 5 points tolerance

    def onpick(event):
        thisline = event.artist
        xdata = thisline.get_xdata()
        ydata = thisline.get_ydata()
        ind = event.ind
        points = tuple(zip(xdata[ind], ydata[ind]))
        print('onpick points:', points)

    fig.canvas.mpl_connect('pick_event', onpick)

    plt.show()

def plotter(data, x, y, y2=None, idx1=0, idx2=-1, dec=1, plot_type="xy", ylog=False, y2log=False):
    """some documentation"""
    if plot_type == "xy":
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        if x != None:
            axs.plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color='mediumblue')
            axs.set_xlabel(x)
        else: 
            axs.plot(data[y][idx1:idx2:dec], color='mediumblue')
            axs.set_xlabel("record #")
        axs.set_ylabel(y)
        if ylog == True:
            axs.set_yscale('log')
        plt.tight_layout()
        plt.show()
    
    elif plot_type == "xyy":
        
        fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        axs2 = axs.twinx()
        if x != None:
            axs.plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color='mediumblue')
            axs2.plot(data[x][idx1:idx2:dec], data[y2][idx1:idx2:dec], color='crimson')
            axs.set_xlabel(x)
        else:
            axs.plot(data[y][idx1:idx2:dec], color='mediumblue')
            axs2.plot(data[y2][idx1:idx2:dec], color='crimson')
            axs.set_xlabel("record #")
        axs.set_ylabel(y +"\n"+ y2)
        if ylog == True:
            axs.set_yscale('log')
        if y2log == True:
            axs2.set_yscale('log')
        plt.tight_layout()
        plt.show()

    elif plot_type == "sub":
        
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        if x != None:
            axs[0].plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color='mediumblue')
            axs[1].plot(data[x][idx1:idx2:dec], data[y2][idx1:idx2:dec], color='crimson')
            axs[1].set_xlabel(x)
        else:
            axs[0].plot(data[y][idx1:idx2:dec], color='mediumblue')
            axs[1].plot(data[y2][idx1:idx2:dec], color='crimson')
            axs[1].set_xlabel(x)
        axs[0].set_ylabel(y)
        axs[1].set_ylabel(y2)
        if ylog == True:
            axs[0].set_yscale('log')
        if y2log == True:
            axs[1].set_yscale('log')
        plt.tight_layout()
        plt.show()



def plotfr(data, time, y, idx1, idx2, Fs, maglog=False, showphase=False):

    # dt = np.diff(data[time][idx1:idx2])[0]
    # Fs = 1 / dt  # sampling frequency

    
    fig = plt.figure(figsize=(8,12))
    # plot time signal:
    ax1 = fig.add_subplot(311)
    ax1.set_title("Signal")
    ax1.plot(data[time][idx1:idx2], data[y][idx1:idx2], color='mediumblue')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Amplitude")

    # plot different spectrum types:
    ax2 = fig.add_subplot(312)
    if maglog != True:
        ax2.set_title("Magnitude Spectrum")
        ax2.magnitude_spectrum(data[y][idx1:idx2], Fs=Fs, color='darkgreen')
    else:
        ax2.set_title("Log. Magnitude Spectrum")
        ax2.magnitude_spectrum(data[y][idx1:idx2], Fs=Fs, scale='dB', color='darkgreen')

    if showphase == True:
        ax3 = fig.add_subplot(313, sharex=ax2)
        ax3.set_title("Phase Spectrum ")
        ax3.phase_spectrum(data[y][idx1:idx2], Fs=Fs, color='orangered')

    plt.tight_layout()
    plt.show()

