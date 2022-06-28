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


def plotter(data, x, y, y2=None, y3=None, idx1=0, idx2=-1, dec=1, plot_type="xy", ylog=False, y2log=False, y3log=False):
    """some documentation"""

    # Plotting Settings:
    # -------------------------------------------------------------------------------------------------------- 
    color1 = 'mediumblue'
    color2 = 'crimson' 
    color3 = 'black' 
    color4 = 'darkgreen'
    textsize1 = 16
    textsize2 = 20
    axis_font = {'size':textsize2}

    font = {'family': 'Arial',
            # 'weight': 'bold',
            'size': textsize1}
    
    mpl.rc('font', **font)

    # Single XY Plot: 
    # --------------------------------------------------------------------------------------------------------
    if plot_type == "xy":
        fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        if x != None:
            axs.plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color=color1)
            axs.set_xlabel(x, **axis_font)
        else: 
            axs.plot(data[y][idx1:idx2:dec], color=color1)
            axs.set_xlabel("record #", **axis_font)
        axs.set_ylabel(y, **axis_font)
        if ylog == True:
            axs.set_yscale('log')
        plt.tight_layout()
        plt.show()
    
    # Plot With Two Different Y Axis:
    # --------------------------------------------------------------------------------------------------------
    elif plot_type == "xyy":
        fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True)
        axs2 = axs.twinx()
        if x != None:
            axs.plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color=color1)
            axs.set_ylabel(y, color=color1, **axis_font)
            axs.set_xlabel(x, **axis_font)
            
            axs2.plot(data[x][idx1:idx2:dec], data[y2][idx1:idx2:dec], color=color2)
            axs2.set_ylabel(y2, color=color2, **axis_font)
       
        else:
            axs.plot(data[y][idx1:idx2:dec], color=color1)
            axs.set_ylabel(y, color1, **axis_font)
            axs.set_xlabel("record #", **axis_font)

            axs2.plot(data[y2][idx1:idx2:dec], color=color2)
            axs2.set_ylabel(y2, color2, **axis_font)
        
        if ylog == True:
            axs.set_yscale('log')
        if y2log == True:
            axs2.set_yscale('log')
        plt.tight_layout()
        plt.show()

    # Plot Which Shares Y Axis: 
    # --------------------------------------------------------------------------------------------------------
    elif plot_type == "sharey":
        fig, axs = plt.subplots(1, 1, figsize=(10, 8), sharex=True, sharey=True)

        if x != None:
            axs.plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color=color1, label=y)
            axs.plot(data[x][idx1:idx2:dec], data[y2][idx1:idx2:dec], color=color2, label=y2)
            axs.set_ylabel(y, **axis_font)
            axs.set_xlabel(x, **axis_font)

        else:
            axs.plot(data[y][idx1:idx2:dec], color=color1, label=y)
            axs.plot(data[y2][idx1:idx2:dec], color=color2, label=y2)
            axs.set_ylabel(y, **axis_font)
            axs.set_xlabel("record #", **axis_font)

        if ylog == True:
            axs.set_yscale('log')

        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()

    
    # Subplots Where Two Plots are Stacked:
    # --------------------------------------------------------------------------------------------------------
    elif plot_type == "sub2":
        fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        if x != None:
            axs[0].plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color=color1)
            axs[0].set_ylabel(y, **axis_font)
            
            axs[1].plot(data[x][idx1:idx2:dec], data[y2][idx1:idx2:dec], color=color2)
            axs[1].set_ylabel(y2, **axis_font)
            axs[1].set_xlabel(x, **axis_font)
        
        else:
            axs[0].plot(data[y][idx1:idx2:dec], color=color1)
            axs[0].set_ylabel(y, **axis_font)
            
            axs[1].plot(data[y2][idx1:idx2:dec], color=color2)
            axs[1].set_ylabel(y2, **axis_font)
            axs[1].set_xlabel(x, **axis_font)
        
        if ylog == True:
            axs[0].set_yscale('log')
        if y2log == True:
            axs[1].set_yscale('log')
        
        plt.tight_layout()
        plt.show()

    # Subplots Where Three Plots are Stacked:
    # --------------------------------------------------------------------------------------------------------
    elif plot_type == "sub3":
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        if x != None:
            axs[0].plot(data[x][idx1:idx2:dec], data[y][idx1:idx2:dec], color=color1)
            axs[0].set_ylabel(y, **axis_font)

            axs[1].plot(data[x][idx1:idx2:dec], data[y2][idx1:idx2:dec], color=color2)
            axs[1].set_ylabel(y2, **axis_font)

            axs[2].plot(data[x][idx1:idx2:dec], data[y3][idx1:idx2:dec], color=color3)
            axs[2].set_ylabel(y3, **axis_font)
            axs[2].set_xlabel(x, **axis_font)
        
        else:
            axs[0].plot(data[y][idx1:idx2:dec], color=color1)
            axs[0].set_ylabel(y, **axis_font)

            axs[1].plot(data[y2][idx1:idx2:dec], color=color2)
            axs[1].set_ylabel(y2, **axis_font)

            axs[2].plot(data[y3][idx1:idx2:dec], color=color3)
            axs[2].set_ylabel(y3, **axis_font)
            axs[1].set_xlabel(x, **axis_font)
        
        if ylog == True:
            axs[0].set_yscale('log')
        if y2log == True:
            axs[1].set_yscale('log')
        if y3log == True:
            axs[2].set_yscale('log')

        plt.tight_layout()
        plt.show()


def plotfr(data, x, y, Fs, idx1, idx2, maglog=False, showphase=False):

    # Plotting Settings:
    # -------------------------------------------------------------------------------------------------------- 
    color1 = 'mediumblue'
    color2 = 'crimson' 
    color3 = 'black' 
    color4 = 'darkgreen'
    textsize1 = 16
    textsize2 = 20
    axis_font = {'size':textsize2}

    font = {'family': 'Arial',
            # 'weight': 'bold',
            'size': textsize1}
    
    mpl.rc('font', **font)

    fig = plt.figure(figsize=(8,12))
    # plot time signal:
    ax1 = fig.add_subplot(311)
    ax1.set_title("Signal")
    ax1.plot(data[x][idx1:idx2], data[y][idx1:idx2], color=color1)
    ax1.set_xlabel(x, **axis_font)
    ax1.set_ylabel("Amplitude", **axis_font)

    # plot different spectrum types:
    ax2 = fig.add_subplot(312)
    if maglog != True:
        ax2.set_title("Magnitude Spectrum")
        ax2.magnitude_spectrum(data[y][idx1:idx2], Fs=Fs, color=color2)
    else:
        ax2.set_title("Log. Magnitude Spectrum")
        ax2.magnitude_spectrum(data[y][idx1:idx2], Fs=Fs, scale='dB', color=color2)

    if showphase == True:
        ax3 = fig.add_subplot(313, sharex=ax2)
        ax3.set_title("Phase Spectrum ")
        ax3.phase_spectrum(data[y][idx1:idx2], Fs=Fs, color=color4)

    plt.tight_layout()
    plt.show()

