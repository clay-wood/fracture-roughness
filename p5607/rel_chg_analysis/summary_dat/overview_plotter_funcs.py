import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
# from plotter_functions import *
from legend_specs import *
#----------------------------------------------------------------


def relChg(data, x, y1, xlim, ylim, **kwargs):
    mpl.rcParams.update({'font.size': 12})
    
    colors = ['#fcbba1', '#fb6a4a', '#a50f15', '#c6dbef', '#6baed6', '#08519c', '#006d2c']
        
    ns_runs = ['$\sigma_{eff}$ = 10 MPa', '$\sigma_{eff}$ = 12.5 MPa', '$\sigma_{eff}$ = 15 MPa', '$\sigma_{eff}$ = 17.5 MPa', '$\sigma_{eff}$ = 20 MPa']

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 5, gridspec_kw={'wspace': 0.05})

    amps = np.unique(np.round(np.hstack(data[x,0:2]),1))
    
    for aa in range(5):
        for bb in range(7):
            for amp in amps:
                Ymean = np.nanmean(np.hstack(data[y1][(aa*2):(aa*2)+2])[bb][np.round(np.hstack(data[x,(aa*2):(aa*2)+2]),1)==amp])
                Yerr = np.nanstd(np.hstack(data[y1][(aa*2):(aa*2)+2])[bb][np.round(np.hstack(data[x,(aa*2):(aa*2)+2]),1)==amp])
                axs[aa].errorbar(amp, Ymean, yerr=Yerr, fmt = 'o', markersize=6, mfc=colors[bb], ecolor=colors[bb])
        axs[aa].set_title(ns_runs[aa])
        # axs[aa].plot([0, 0], [-1, 1], color = 'gray',ls = '--') #Vertical Line
        axs[aa].plot([-2, 2], [0,0], color = 'gray',ls = '--') #Horizontal Line
    
    # AXES: MINOR TICKS, COLORS 
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(kwargs['xlabel'],color='black')
    for ax in axs[1::]:
        ax.tick_params(labelleft=False)
 
    # AXES LABELS & COLORS   
    axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs[0].set_ylabel(kwargs['ylabel'], color='black')
    plt.show()
    if kwargs['filename'] != '': 
        fig.savefig(kwargs['filename']+'.svg')
        fig.savefig(kwargs['filename']+'.pdf')

#----------------------------------------------------------------
#----------------------------------------------------------------

def slope(data, x, y1, xlim, ylim, **kwargs):
    mpl.rcParams.update({'font.size': 12})

    colors = ['#fcbba1', '#fb6a4a', '#a50f15', '#c6dbef', '#6baed6', '#08519c', '#006d2c']

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 1, figsize=(3,3))

    nsLevels = np.array([10,12.5,15,17.5,20])
    slopes = np.zeros((5,7,2))
    
    for aa in range(5):
        for bb in range(7):
            slopes[aa,bb,:] = np.polyfit(np.hstack(data[x,(aa*2):(aa*2)+2]), np.hstack(data[y1][(aa*2):(aa*2)+2])[bb], 1)
    for bb in range(7):
        axs.errorbar(nsLevels, slopes[:,bb,0], yerr=slopes[:,bb,1], fmt = 'o', markersize=6, mfc=colors[bb], ecolor=colors[bb])
        axs.plot(nsLevels, slopes[:,bb,0], color=colors[bb],ls='--')
        axs.plot([5, 25], [0,0], color = 'gray',ls = '--') #Horizontal Line
    
    # AXES: MINOR TICKS, COLORS 
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.set_xlabel(kwargs['xlabel'],color='black')
 
    # AXES LABELS & COLORS   
    axs.ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs.set_ylabel(kwargs['ylabel'], color='black')
    axs.set_xticks(np.arange(10, 22.5, 2.5))
    plt.show()
    if kwargs['filename'] != '': 
        fig.savefig(kwargs['filename']+'.svg')
        fig.savefig(kwargs['filename']+'.pdf')

#----------------------------------------------------------------
#----------------------------------------------------------------

def relChg_perm(data, x, y1, xlim, ylim, **kwargs):
    mpl.rcParams.update({'font.size': 12})
        
    ns_runs = ['$\sigma_{eff}$ = 10 MPa', '$\sigma_{eff}$ = 12.5 MPa', '$\sigma_{eff}$ = 15 MPa', '$\sigma_{eff}$ = 17.5 MPa', '$\sigma_{eff}$ = 20 MPa']

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 5, gridspec_kw={'wspace': 0.05})

    amps = np.unique(np.round(np.hstack(data[x,0:2]),1))
    
    for aa in range(5):
        for amp in amps:
            Ymean = np.nanmean(np.hstack(data[y1][(aa*2):(aa*2)+2])[np.round(np.hstack(data[x,(aa*2):(aa*2)+2]),1)==amp])
            Yerr = np.nanstd(np.hstack(data[y1][(aa*2):(aa*2)+2])[np.round(np.hstack(data[x,(aa*2):(aa*2)+2]),1)==amp])
            axs[aa].errorbar(amp, Ymean, yerr=Yerr, fmt = 'o', markersize=6, mfc='k', ecolor='k')
        # axs[aa].set_yscale('symlog')
        axs[aa].set_title(ns_runs[aa])
        # axs[aa].plot([0, 0], [-1, 1], color = 'gray',ls = '--') #Vertical Line
        axs[aa].plot([-2, 2], [0,0], color = 'k',ls = '--') #Horizontal Line
    
    # AXES: MINOR TICKS, COLORS 
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(kwargs['xlabel'],color='black')
    for ax in axs[1::]:
        ax.tick_params(labelleft=False)
 
    # AXES LABELS & COLORS   
    # axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs[0].set_ylabel(kwargs['ylabel'], color='black')
    plt.show()
    if kwargs['filename'] != '': 
        fig.savefig(kwargs['filename']+'.svg')
        fig.savefig(kwargs['filename']+'.pdf')

#----------------------------------------------------------------
#----------------------------------------------------------------

def slope_perm(data, x, y1, xlim, ylim, **kwargs):
    from sklearn.metrics import r2_score

    mpl.rcParams.update({'font.size': 12})

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 1, figsize=(3,3))

    nsLevels = np.array([10,12.5,15,17.5,20])
    slopes = np.zeros((5,3))

    for aa in range(5):
        fin_idx = np.isfinite(np.hstack(data[y1][(aa*2):(aa*2)+2]))
        slopes[aa,0:2] = np.polyfit(np.hstack(data[x,(aa*2):(aa*2)+2])[fin_idx], np.hstack(data[y1][(aa*2):(aa*2)+2])[fin_idx], 1)
        Q = np.polyval(slopes[aa,0:2], np.hstack(data[x,(aa*2):(aa*2)+2]))
        slopes[aa,2] = r2_score(np.hstack(data[y1][(aa*2):(aa*2)+2])[fin_idx], Q[fin_idx])

    axs.errorbar(nsLevels, slopes[:,0], yerr=slopes[:,1], fmt = 'o', color='k', markersize=6, mfc='k', ecolor='k')
    axs.set_yscale('log')
    
    # AXES: MINOR TICKS, COLORS 
    axs.set_xlim(xlim)
    axs.set_ylim(ylim)
    axs.set_xlabel(kwargs['xlabel'],color='black')
 
    # AXES LABELS & COLORS   
    # axs.ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs.set_ylabel(kwargs['ylabel'], color='black')
    axs.set_xticks(np.arange(10, 22.5, 2.5))
    plt.show()
    if kwargs['filename'] != '': 
        fig.savefig(kwargs['filename']+'.svg')
        fig.savefig(kwargs['filename']+'.pdf')

#----------------------------------------------------------------
#----------------------------------------------------------------

def relChgLog_perm(data, x, y1, xlim, ylim, **kwargs):
    mpl.rcParams.update({'font.size': 12})
        
    ns_runs = ['$\sigma_{eff}$ = 10 MPa', '$\sigma_{eff}$ = 12.5 MPa', '$\sigma_{eff}$ = 15 MPa', '$\sigma_{eff}$ = 17.5 MPa', '$\sigma_{eff}$ = 20 MPa']

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 5, gridspec_kw={'wspace': 0.05})

    amps = np.unique(np.round(np.hstack(data[x,0:2]),1))
    
    for aa in range(5):
        for amp in amps:
            Ymean = np.nanmean(np.hstack(data[y1][(aa*2):(aa*2)+2])[np.round(np.hstack(data[x,(aa*2):(aa*2)+2]),1)==amp])
            Yerr = np.nanstd(np.hstack(data[y1][(aa*2):(aa*2)+2])[np.round(np.hstack(data[x,(aa*2):(aa*2)+2]),1)==amp])/np.log(10)
            asymYerr = [Ymean-Yerr, Ymean+Yerr]
            axs[aa].errorbar(amp, Ymean, yerr=Yerr, fmt = 'o', markersize=6, mfc='k', ecolor='k')
        axs[aa].set_title(ns_runs[aa])
    
    # AXES: MINOR TICKS, COLORS 
    for ax in axs:
        ax.set_yscale('log', nonpositive="clip")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(kwargs['xlabel'],color='black')
    for ax in axs[1::]:
        ax.tick_params(labelleft=False)
 
    # AXES LABELS & COLORS   
    # axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs[0].set_ylabel(kwargs['ylabel'], color='black')
    plt.show()
    if kwargs['filename'] != '': 
        fig.savefig(kwargs['filename']+'.svg')
        fig.savefig(kwargs['filename']+'.pdf')

#----------------------------------------------------------------
#----------------------------------------------------------------

def relChgBoth(data, x1, x2, y1, xlim, ylim, **kwargs):
    mpl.rcParams.update({'font.size': 12})
    
    colors = ['#fcbba1', '#fb6a4a', '#a50f15', '#c6dbef', '#6baed6', '#08519c', '#006d2c']
        
    ns_runs = ['$\sigma_{eff}$ = 10 MPa', '$\sigma_{eff}$ = 12.5 MPa', '$\sigma_{eff}$ = 15 MPa', '$\sigma_{eff}$ = 17.5 MPa', '$\sigma_{eff}$ = 20 MPa']

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 5, gridspec_kw={'wspace': 0.05})

    amps = np.unique(np.round(np.hstack(data[x1,0:2]),1))
    
    for aa in range(5):
        for amp in amps:
            Ymean = np.nanmean(np.hstack(data[y1][(aa*2):(aa*2)+2])[:,np.round(np.hstack(data[x1,(aa*2):(aa*2)+2]),1)==amp])
            Yerr = np.nanmean(np.hstack(data[y1][(aa*2):(aa*2)+2])[:,np.round(np.hstack(data[x1,(aa*2):(aa*2)+2]),1)==amp])

            Xmean = np.nanmean(np.hstack(data[x2][(aa*2):(aa*2)+2])[np.round(np.hstack(data[x1,(aa*2):(aa*2)+2]),1)==amp])
            Xerr = np.nanstd(np.hstack(data[x2][(aa*2):(aa*2)+2])[np.round(np.hstack(data[x1,(aa*2):(aa*2)+2]),1)==amp])

            axs[aa].errorbar(Xmean, Ymean, xerr=Xerr, yerr=Yerr, fmt = 'o', markersize=6, mfc='k', ecolor='k')
            axs[aa].set_title(ns_runs[aa])
        axs[aa].plot([0, 0], [-1, 1], color = 'gray',ls = '--') #Vertical Line
        axs[aa].plot([-150, 150], [0,0], color = 'gray',ls = '--') #Horizontal Line
    
    # AXES: MINOR TICKS, COLORS 
    for ax in axs:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(kwargs['xlabel'],color='black')
    for ax in axs[1::]:
        ax.tick_params(labelleft=False)
 
    # AXES LABELS & COLORS   
    axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs[0].set_ylabel(kwargs['ylabel'], color='black')
    plt.show()
    if kwargs['filename'] != '': 
        fig.savefig(kwargs['filename']+'.svg')
        fig.savefig(kwargs['filename']+'.pdf')

#----------------------------------------------------------------
#----------------------------------------------------------------

def slopeBoth(data, x1, x2, y1, xlim, ylim, **kwargs):

    mpl.rcParams.update({'font.size': 12})

    plt.style.use('jgr2020_style3.mplstyle')
    fig, axs = plt.subplots(1, 1, figsize=(3,3))

    nsLevels = np.array([10,12.5,15,17.5,20])
    slopes = np.zeros((5,7,3))

    for aa in range(5):
        fin_idx = np.isfinite(np.hstack(data[x2][(aa*2):(aa*2)+2]))
        for bb in range(7):
            slopes[aa,bb,0:2] = np.polyfit(np.hstack(data[x2,(aa*2):(aa*2)+2])[fin_idx], np.hstack(data[y1][(aa*2):(aa*2)+2])[bb,fin_idx], 1)
    Ymean = 1e4*np.mean(slopes, 1)[:,0]
    Yerr = 1e4*np.std(slopes, 1)[:,0]

    axs.errorbar(nsLevels, Ymean, yerr=Yerr, fmt = 'o', color='k', markersize=6, mfc='k', ecolor='k')
    
    # AXES: MINOR TICKS, COLORS 
    axs.set_xlim(xlim)
    # axs.set_ylim(ylim)
    axs.set_xlabel(kwargs['xlabel'],color='black')
 
    # AXES LABELS & COLORS   
    axs.ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
    axs.set_ylabel(kwargs['ylabel'], color='black')
    axs.set_xticks(np.arange(10, 22.5, 2.5))
    plt.show()


# def relChg_amp_plots2(data, x, y1, y2, color1, color2, xlim, ylim, xlabel, ylabel):
    
#     [legend_elements1N, legend_elements1P, legend_elements2N, legend_elements2P, legend_elements2Q, legend_elements2R] = legend_stuff(color1,color2)
#     mpl.rcParams.update({'font.size': 12})
    
# #     colors = ['dodgerblue', 'mediumblue', 'navy', 'pink', 'deeppink', 'mediumvioletred', 'goldenrod']
#     colors = ['#fcbba1', '#fb6a4a', '#a50f15', '#c6dbef', '#6baed6', '#08519c', '#006d2c']
#     mrkr = ['o', 's', '^', 'd', '*']
#     sz = [6, 6, 7, 7, 9]
        
# #     ns_runs = ['$\sigma_{eff}$ = 10 MPa', '$\sigma_{eff}$ = 12.5 MPa', '$\sigma_{eff}$ = 15 MPa', '$\sigma_{eff}$ = 17.5 MPa', '$\sigma_{eff}$ = 20 MPa']

#     plt.style.use('jgr2020_style3.mplstyle')
#     fig, axs = plt.subplots(1, 7, figsize=(22,3), gridspec_kw={'wspace': 0.05})
    
#     for aa in range(5):
#         for bb in range(7):
#             axs[bb].errorbar(data[x, aa*2], data[y1][aa*2][bb], yerr=None, fmt = mrkr[aa], markersize=sz[aa], mfc=colors[bb])
#             axs[bb].errorbar(data[x, (aa*2)+1], data[y1][(aa*2)+1][bb], yerr=None, fmt=mrkr[aa], markersize=sz[aa], mfc=colors[bb])

# #         axs[aa].set_title(ns_runs[aa])
 
#     for ax in axs:
#         ax.plot([-100, 200], [0, 0], color = 'gray',ls = '--') #Horizontal Line
#         ax.plot([0, 0], [-100, 200], color = 'gray',ls = '--') #Horizontal Line
        
# #     legend1 = plt.legend(handles = legend_elements1P, title='Pair', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #     legend2 = plt.legend(handles = legend_elements2P, title='$Osc.\ Set$', bbox_to_anchor=(1.05, 0.65), loc='upper left', borderaxespad=0.)
       
# #     axs[-1].add_artist(legend1); axs[-1].add_artist(legend2)
    
#     # AXES: MINOR TICKS, COLORS 
#     for ax in axs:
#         ax.set_xlim(xlim)
#         ax.set_ylim(ylim)
#         ax.set_xlabel(xlabel,color='black')
# #         ax.set_xticks(np.arang3e(0.2, 1.2, 0.2))
#     for ax in axs[1::]:
#         ax.tick_params(labelleft=False)
 
#     # AXES LABELS & COLORS   
#     axs[0].ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
#     axs[0].set_ylabel(ylabel, color='black')
#     plt.show()

# #----------------------------------------------------------------
# #----------------------------------------------------------------

# def relChg_amp_plots3(data, x, y1, y2, color1, color2, xlim, ylim, xlabel, ylabel):
    
#     [legend_elements1N, legend_elements1P, legend_elements2N, legend_elements2P, legend_elements2Q, legend_elements2R] = legend_stuff(color1,color2)
#     mpl.rcParams.update({'font.size': 12})
#     mpl.rcParams.update({'font.family':'sans-serif'})
#     mpl.rcParams.update({'font.sans-serif':'Arial'})
    
#     colors = ['#fcbba1', '#fb6a4a', '#a50f15', '#c6dbef', '#6baed6', '#08519c', '#006d2c']
#     mrkr = ['o', 's', '^', 'd', '*']
#     sz = [6, 6, 7, 7, 9]
        
# #     ns_runs = ['$\sigma_{eff}$ = 10 MPa', '$\sigma_{eff}$ = 12.5 MPa', '$\sigma_{eff}$ = 15 MPa', '$\sigma_{eff}$ = 17.5 MPa', '$\sigma_{eff}$ = 20 MPa']

#     plt.style.use('jgr2020_style3.mplstyle')
#     axd = plt.figure(figsize=(10,10)).subplot_mosaic(
#     """
#     XXg
#     def
#     abc
#     """,
#     empty_sentinel="X", gridspec_kw={"width_ratios": [1, 1, 1], 'wspace': 0.06, 'hspace': 0.06})
#     plot_names = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
#     for aa in range(5):
#         for bb in range(7):
#             axd[plot_names[bb]].errorbar(data[x, aa*2], data[y1][aa*2][bb], yerr=None, fmt = mrkr[aa], markersize=sz[aa], mfc=colors[bb])
# #             axd[bb].errorbar(data[x, (aa*2)+1], data[y1][(aa*2)+1][bb], yerr=None, fmt=mrkr[aa], markersize=sz[aa], mfc=colors[bb])
#             axd[plot_names[bb]].plot([-100, 200], [0, 0], color = 'gray',ls = '--') #Horizontal Line
#             axd[plot_names[bb]].plot([0, 0], [-100, 200], color = 'gray',ls = '--') #Vertical Line
#             axd[plot_names[bb]].set_xlim(xlim)
#             axd[plot_names[bb]].set_xticks(np.arange(0.2, 1.2, 0.2))
#             axd[plot_names[bb]].set_ylim(ylim)
#             axd[plot_names[bb]].ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
# #         axs[aa].set_title(ns_runs[aa])
 
        
# #     legend1 = plt.legend(handles = legend_elements1P, title='Pair', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
# #     legend2 = plt.legend(handles = legend_elements2P, title='$Osc.\ Set$', bbox_to_anchor=(1.05, 0.65), loc='upper left', borderaxespad=0.)
       
# #     axs[-1].add_artist(legend1); axs[-1].add_artist(legend2)
    
#     # AXES: MINOR TICKS, COLORS 
#     for no_ytick_label in ['b','c','e','f']:
#         axd[no_ytick_label].tick_params(labelleft=False)
#     for no_xtick_label in ['d', 'e', 'f', 'g']:
#         axd[no_xtick_label].tick_params(labelbottom=False)
 
#     # AXES LABELS & COLORS   
#     for yaxis_label in ['a', 'd', 'g']:
#         axd[yaxis_label].set_ylabel(ylabel, color='black')
#     for xaxis_label in ['a', 'b', 'c']:
#         axd[xaxis_label].set_xlabel(xlabel, color='black')
        
#     plt.show()
#     plt.close()

# #----------------------------------------------------------------
# #----------------------------------------------------------------

# def slope_plots(data, x, y1, y2, color1, color2, xlim, ylim, xlabel, ylabel):
#     [legend_elements1N, legend_elements1P, legend_elements2N, legend_elements2P, legend_elements2Q, legend_elements2R] = legend_stuff(color1,color2)
#     mpl.rcParams.update({'font.size': 12})
#     plt.style.use('jgr2020_style3.mplstyle')
#     fig, axs = plt.subplots(1, 1, figsize=(3,3))
    
#     ns_levels = np.array([10, 12.5, 15, 17.5, 20])
#     q = np.zeros((5,2)); q2 = np.zeros((5,2))
#     err = np.zeros(5); err2 = np.zeros(5)

#     for aa in range(5):
#         fin_idx = np.isfinite(np.hstack(data[(aa*2):(aa*2)+2,y1])) & np.isfinite(np.hstack(data[(aa*2):(aa*2)+2,x]))
#         q[aa,:] = np.polyfit(np.hstack(data[(aa*2):(aa*2)+2,x])[fin_idx], np.hstack(data[(aa*2):(aa*2)+2,y1])[fin_idx], 1)
#         err[aa] = np.std(np.hstack(data[(aa*2):(aa*2)+2,y1])[fin_idx])
#         if y2 != None:
#             fin_idx2 = np.isfinite(np.hstack(data[(aa*2):(aa*2)+2,y2])) & np.isfinite(np.hstack(data[(aa*2):(aa*2)+2,x]))
#             q2[aa,:] = np.polyfit(np.hstack(data[(aa*2):(aa*2)+2,x])[fin_idx2], np.hstack(data[(aa*2):(aa*2)+2,y2])[fin_idx2], 1)
#             err2[aa] = np.std(np.hstack(data[(aa*2):(aa*2)+2,y2])[fin_idx])

#     axs.errorbar(ns_levels, q[:,0], yerr=err, fmt = 'd', markersize=6, mfc=color1)
#     axs.plot(ns_levels, q[:,0], c=color1, ls='--')
#     axs.errorbar(ns_levels, q2[:,0], yerr=err2, fmt = '*', markersize=8, mfc=color2)
#     axs.plot(ns_levels, q2[:,0], c=color2, ls='--')

#     # AXES: MINOR TICKS, COLORS 
#     axs.set_xlim(xlim)
#     axs.set_ylim(ylim)
#     axs.set_xlabel(xlabel,color='black')
#     if x == 0:
#         axs.set_xticks(np.arange(10, 22.5, 2.5))

#     legend1 = plt.legend(handles = legend_elements1P, title='Pair', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
#     # # AXES LABELS & COLORS   
#     axs.ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
#     axs.set_ylabel(ylabel, color='black')
#     plt.show()
    
#     return q, err, q2, err2

# #----------------------------------------------------------------
# #----------------------------------------------------------------

# def slope_plots2(data, x, y1, color1, xlim, ylim, xlabel, ylabel):
# #     [legend_elements1N, legend_elements1P, legend_elements2N, legend_elements2P, legend_elements2Q, legend_elements2R] = legend_stuff(color1,color2)
#     mpl.rcParams.update({'font.size': 12})
#     plt.style.use('jgr2020_style3.mplstyle')
#     fig, axs = plt.subplots(1, 1, figsize=(3,3))
    
#     ns_levels = np.array([10, 12.5, 15, 17.5, 20])
#     q = np.zeros((5,2)); q2 = np.zeros((5,2))
#     err = np.zeros(5); err2 = np.zeros(5)

#     for aa in range(5):
#         fin_idx = np.isfinite(np.hstack(data[(aa*2):(aa*2)+2,y1])) & np.isfinite(np.hstack(data[(aa*2):(aa*2)+2,x]))
#         q[aa,:] = np.polyfit(np.hstack(data[(aa*2):(aa*2)+2,x])[fin_idx], np.hstack(data[(aa*2):(aa*2)+2,y1])[fin_idx], 1)
#         err[aa] = np.std(np.hstack(data[(aa*2):(aa*2)+2,y1])[fin_idx])

#     axs.errorbar(ns_levels, q[:,0], yerr=err, fmt = 'd', markersize=6, mfc=color1, ecolor=color1)
#     axs.plot(ns_levels, q[:,0], c=color1, ls='--')
# #     axs.errorbar(ns_levels, q2[:,0], yerr=err2, fmt = '*', markersize=8, mfc=color2)
# #     axs.plot(ns_levels, q2[:,0], c=color2, ls='--')

#     # AXES: MINOR TICKS, COLORS 
#     axs.set_xlim(xlim)
#     axs.set_ylim(ylim)
#     axs.set_xlabel(xlabel,color='black')
#     if x == 0:
#         axs.set_xticks(np.arange(10, 22.5, 2.5))

# #     legend1 = plt.legend(handles = legend_elements1P, title='Pair', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
#     # # AXES LABELS & COLORS   
#     axs.ticklabel_format(axis='y',style='sci',scilimits=(-4,3))
#     axs.set_ylabel(ylabel, color='black')
#     plt.show()
    
#     return q, err