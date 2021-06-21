import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.metrics import confusion_matrix

from ninolearn.learn.skillMeasures import seasonal_correlation
import matplotlib.pyplot as plt

seismic = plt.cm.get_cmap('seismic', 256)
newcolors = seismic(np.linspace(0, 1, 256))
grey = np.array([192/256, 192/256, 192/256, 1])
newcolors[:1, :] = grey
newcmp = ListedColormap(newcolors)

seas_ticks = ['DJF', 'JFM', 'FMA', 'MAM', 'AMJ', 'MJJ',
                        'JJA', 'JAS', 'ASO', 'SON', 'OND', 'NDJ']

mon_ticks = ['J', 'F', 'M', 'A', 'M', 'J',
                        'J', 'A', 'S', 'O', 'N', 'D']

def plot_correlation(y, pred, time, title=None):
    """
    make a bar plot of the correlation coeficent between y and the prediction
    """
    m = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(5,2.5))

    r, p = seasonal_correlation(y, pred, time)

    ax.set_ylim(0, 1)
    ax.bar(m, r)
    ax.set_xticks(m)
    ax.set_xticklabels(seas_ticks)
    ax.set_xlabel("Season")
    ax.set_ylabel(f"Correlation coefficient")
    if title is None:
        ax.set_title(f"$r =$ {round(np.corrcoef(y,pred)[0,1], 2)}")
    else:
         ax.set_title(title)
    plt.tight_layout()

def plot_confMat(y, pred, labels):
    """
    Plot a confusion matrix. Here, the recall is on the diagonal!

    :param y: The baseline.
    :param pred: The prediction.
    :param labels: The names of the classes.
    """
    cm = confusion_matrix(y, pred)#.T
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues,
                   vmin = 1/len(labels), vmax = 0.8)
    ax.figure.colorbar(im, ax=ax,extend='max')
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels,
           yticklabels=labels,
           title='Confusion Matrix',
           xlabel='True label',
           ylabel='Predicted label')

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="black" if cm[i, j] > thresh else "black")
    fig.tight_layout()


def plot_seasonal_skill(lead_time, data, vmin=-1, vmax=1, nlevels=20, cmap=newcmp, extend='min'):
    fig, ax = plt.subplots(figsize=(5,3.5))
    m = np.arange(1,13)

    levels = np.linspace(vmin, vmax, nlevels+1)
    C = ax.contourf(m,lead_time, data, levels=levels,
                 vmin=vmin, vmax=vmax,
                 cmap=cmap, extend=extend)

    ax.set_xticks(m)
    ax.set_xticklabels(seas_ticks, rotation='vertical')
    ax.set_xlabel('Target Season')
    ax.set_yticks(lead_time)
    ax.set_yticklabels(lead_time)
    ax.set_ylabel('Lead Time [Months]')
    plt.colorbar(C, ticks=np.arange(vmin,vmax+0.1,0.2))
    plt.tight_layout()

from ninolearn.learn.fit import decade_name

def plot_seasonal_skill_ZC(lead_time, data, vmin=-1, vmax=1, nlevels=20, cmap=newcmp, extend='min'):
    """
    
    """
    fig, ax = plt.subplots(figsize=(5,3.5))
    m = np.arange(1,5)

    levels = np.linspace(vmin, vmax, nlevels+1)
    C = ax.contourf(m,lead_time, data, levels=levels,
                vmin=vmin, vmax=vmax,
                 cmap=cmap, extend=extend)
    D = ax.contour(m, lead_time, data, levels = [-0.9, -0.7, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9], colors = ['black'], linestyles = ['dashed', 'solid'])
    ax.clabel(D, inline = True, fontsize = 10)  
    
    ax.set_xticks(m)
    ax.set_xticklabels(decade_name, rotation=30)
    ax.set_xlabel('Target Decade') ### <-- TODO: make this refer to the decade being predicted 
    ax.set_yticks(lead_time)
    ax.set_yticklabels(lead_time)
    ax.set_ylabel('Lead Time [Months]')
    plt.colorbar(C, ticks=np.arange(vmin,vmax+0.1,0.2))
    plt.tight_layout()
    
from os.path import join    
from ninolearn.private import plotdir
    

def ACC_skill_comparison_ZC(r, rref, lead_times, train_version, test_version, plot_individual = True, plot_avg = False):
    """
    plots ACC prediction skill as function of lead time for several decades. 
    
    r, rref: correlations between obs and predictions for lead times (axis=0) and decades (axis=1)
    """
    
    colors = ['red', 'green', 'blue', 'purple']
    plt.title('ACC skill comparison')
    
    if plot_individual == True: 
        for i in range(rref.shape[1]):    
            plt.plot(lead_times, rref[:,i], label = ('reference (' + str(i+1) + ')'), color = colors[i])
        for j in range(r.shape[1]):
            plt.plot(lead_times, r[:,j], label = ('distorted (' + str(j+1) + ')'), ls='--', color = colors [j])
    if plot_avg == True: 
        r_avg = np.mean(r, axis = 1)
        rref_avg = np.mean(rref, axis =1)
        plt.plot(lead_times, rref_avg, label = 'reference average')
        plt.plot(lead_times, r_avg, label = 'distorted average', ls = '--')
        
    
    plt.xticks(lead_times)
    plt.xlabel('Lead time (months)')
    plt.yticks(np.linspace(0,1, 6))
    plt.ylabel('correlation')
    # plt.hlines(0.6, -1, 22, ls = 'dotted', color = 'grey', label = 'skilful prediction')
    plt.grid()
    
    plt.legend()
    
    plt.savefig(join(plotdir, 'ACC_skill_comparison ' + train_version + '_' + test_version), dpi= 600)
    