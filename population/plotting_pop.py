import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from math import ceil, sqrt, log

hh_2006_data = {}
hh_2006_data['couple_kids'] = [3.34900107726146, 39.0447660216951, 65.3322624325251, 54.7447775052008, 24.9479982067538, 10.6951330955173, 5.22968378587565]
hh_2006_data['couple_only'] = [8.2679400663007, 26.9576962554435, 11.3526674531167, 22.4906274746513, 53.4424459502914, 63.2728988556796, 48.7121992189306]
hh_2006_data['single_kids'] = [1.70297220931839, 6.09711480137245, 9.18963007205741, 8.94723143630836, 4.75604575125046, 3.89202223531223, 6.63623819962609]
hh_2006_data['single_only'] = [3.39125070554798, 9.52051171221026, 8.5740466658523, 10.5389086707166, 15.1736796520539, 21.717575992481, 39.385490518887]
hh_2006_data['with_parents'] = [83.2888359415715, 18.3799112092788, 5.55139337644851, 3.27845491312301, 1.67983043965046, 0.42236982100987, 0.03638827668072]

data_years = [1911, 1921, 1933, 1947, 1954, 1961, 1966, 1971, 1976, 1981, 1986, 1991, 1996, 2001, 2006]
data_sizes = [4.5, 4.4, 4, 3.8, 3.6, 3.6, 3.5, 3.3, 3.1, 3, 2.9, 2.8, 2.6, 2.6, 2.6]
data_counts = [894, 1107, 1510, 1874, 2343, 2782, 3155, 3671, 4141, 4668, 5187, 5750, 6421, 7072, 7596]
data_pop_sizes = [4425000, 5411000, 6603000, 7517000, 8902000, 10391000, 11505000, 12663000, 13892000, 14695000, 15788000, 17065000, 18071000, 19153000, 20697000]

comp_years = [1981, 2001]
comp_props = [[18.0, 24.6], [29.2, 33.3], [16.9, 16.2], [19.1, 16.0], [10.5, 7.3], [6.4, 3.3]]

def plot_hh_type(hh_stats, errors=None):
    """
    Plot distributions of household type by age category.
    """

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    
    fig_titles = {'couple_kids':'Couple with children',
            'couple_only':'Couple without children',
            'single_kids':'Single parent',
            'single_only':'Lone person',
            'with_parents':'With parents'}
    plot_locs = {'couple_kids':321, 'couple_only':322, 'single_kids':323,
            'single_only':324, 'with_parents':325}
    xticl = [' ','15-24','25-34','35-44','45-54','55-64','65-74','75+']
    axis_fontsize = 'small'

    for k in hh_stats:
        bin_count = len(hh_stats[k])
        ax = fig.add_subplot(plot_locs[k])
        if errors:
            ax.bar(range(bin_count), hh_stats[k], width=0.25, color='w',
                    yerr=errors[k], ecolor='k')
        else:
            ax.bar(range(bin_count), hh_stats[k], width=0.25, color='w')
        ax.bar(np.arange(bin_count)-0.25, hh_2006_data[k],
                width=0.25, color='k')
        ax.set_xlim(-0.5,6.5)
        ax.set_ylim(ymin=0, ymax=85)
        ax.set_title(fig_titles[k], fontsize='small')
        ax.axes.get_xaxis().set_ticklabels(xticl)
        for label in ax.axes.get_xaxis().get_ticklabels():
            label.set_fontsize('xx-small')
        for label in ax.axes.get_yaxis().get_ticklabels():
            label.set_fontsize(axis_fontsize)

    return fig


def plot_age_dist(axes, n_means, n_stdev, comp_dat=None):
    """
    Plot age distribution of population.

    :param data: Fraction of individuals by year of age.
    """

    legend_lines = []
    legend_labels = []
    if comp_dat is not None:
        pb1 = axes.plot(range(len(comp_dat)), comp_dat, color='b')
        legend_lines.append(pb1[0])
        legend_labels.append('initial')
    pb2 = axes.errorbar(np.arange(len(n_means)), n_means, n_stdev, color='g')
    legend_lines.append(pb2[0])
    legend_labels.append('final')
#        legend_labels.append('mean and SD final %d years' % samples)

    axes.set_xlabel('Age')
    axes.set_ylabel('Fraction of population')
    axes.set_xlim(xmax=100)
    axes.set_ylim(ymin=0.0)
#    axes.set_title('Age distribution')
    leg = axes.legend(legend_lines, legend_labels)
    leg.get_frame().set_alpha(0.0)


def plot_age_dist_multi(axes, data, comp_dat):
    legend_lines = []
    legend_labels = []
    x = None
    for d in data:
        x = axes.plot(range(len(d)), d, color='0.4')
    d = axes.plot(range(len(comp_dat)), comp_dat, color='k', lw=2)
    legend_lines.append(d[0])
    legend_labels.append('initial')
    legend_lines.append(x[0])
    legend_labels.append('final')
    axes.set_xlabel('Age')
    axes.set_ylabel('Fraction of population')
    axes.set_xlim(xmax=100)
    axes.set_ylim(ymin=0.0)
    leg = axes.legend(legend_lines, legend_labels)
    leg.get_frame().set_alpha(0.0)

   

def plot_hh_change_dist(axes, hh_means, hh_stdev, max_size):

    comp_dat = [0.36666666666667, 0.14041095890411, -0.0414201183432, -0.16230366492147, -0.3047619047619, -0.484375]

    legend_lines = []
    legend_labels = []
    if comp_dat is not None:    # plot data version
        upper = min(len(comp_dat), max_size+1)
        pb1 = axes.bar(np.arange(1,upper+1)-0.25, comp_dat[:upper],
                width=0.25, color='k')
        legend_lines.append(pb1[0])
        legend_labels.append('empirical')
    pb2 = axes.bar(np.arange(1,max_size+1), hh_means[:max_size], 
            width=0.25, yerr=hh_stdev[:max_size], color='w', ecolor='k')
    legend_lines.append(pb2[0])
    legend_labels.append('simulated')
    axes.get_xaxis().set_ticks(range(1,max_size+1))
    axes.set_xlabel('Household size')
    axes.set_ylabel('\% change over final 20 years')
    axes.set_xlim((0.5,max_size+0.5))
#    axes.set_ylim()
#    axes.set_title('Household size distribution')
    leg = axes.legend(legend_lines, legend_labels)
    leg.get_frame().set_alpha(0.0)


def plot_hh_size_dist(axes, hh_means, hh_stdev, max_size, comp_dat=None, comp_label='initial'):
    """
    Plot household size distribution.

    :param data: Fraction of households by size.
    :param max_size: Maximum household size to plot
    """

    legend_lines = []
    legend_labels = []
    if comp_dat is not None:    # plot data version
        upper = min(len(comp_dat), max_size)
        pb1 = axes.bar(np.arange(1,upper)-0.25, comp_dat[1:upper],
                width=0.25, color='k')
        legend_lines.append(pb1[0])
        legend_labels.append(comp_label)
    pb2 = axes.bar(np.arange(1,max_size+1), hh_means[:max_size], 
            width=0.25, yerr=hh_stdev[:max_size], color='w', ecolor='k')
    legend_lines.append(pb2[0])
    legend_labels.append('final')
#    legend_labels.append('mean and SD over %d timesteps' % samples)

    axes.get_xaxis().set_ticks(range(1,max_size+1))
    axes.set_xlabel('Household size')
    axes.set_ylabel('Fraction of households')
    axes.set_xlim((0.5,max_size+0.5))
    axes.set_ylim(ymin=0)
#    axes.set_title('Household size distribution')
    leg = axes.legend(legend_lines, legend_labels)
    leg.get_frame().set_alpha(0.0)


def plot_hh_size_avg(ax, sizes):
    """
    Plot the average household size over time.
    NB: this is currently very specific to case study 3!!!
    """
    p1 = ax.plot(data_years, data_sizes, color='b', 
            marker='o', markerfacecolor='b')
    p2 = ax.plot(range(1910, 1910+len(sizes[100:])), sizes[100:], color='g')
    leg_lines = []
    leg_labels = []
    leg_lines.append(p1[0])
    leg_labels.append('empirical')
    leg_lines.append(p2[0])
    leg_labels.append('simulated')
    ax.set_xlabel('Years')
    ax.set_ylabel('Average household size')
    ax.set_ylim(ymin=0, ymax=5)
    leg = ax.legend(leg_lines, leg_labels)
    leg.get_frame().set_alpha(0.0)


def plot_avg_hh_multi(ax, data):
    for d in data:
        ax.plot(range(1910, 1910+len(d)), d, color='0.4')
    ax.plot(data_years, data_sizes, color='k', marker='o', markerfacecolor='k') 
    ax.set_xlabel('Years')
    ax.set_ylabel('Average household size')
    ax.set_ylim(ymin=0, ymax=5)


def plot_hh_count(ax, counts):
    """
    Plot the household count over time.
    NB: this is currently very specific to case study 3!!!
    """
    scale_factor = float(counts[100]) / data_counts[0]
    scaled_counts = np.array(data_counts) * scale_factor
    p1 = ax.plot(data_years, scaled_counts, color='b', 
            marker='o', markerfacecolor='b')
    p2 = ax.plot(range(1910, 1910+len(counts[100:])), counts[100:], color='g')
    leg_lines = []
    leg_labels = []
    leg_lines.append(p1[0])
    leg_labels.append('empirical')
    leg_lines.append(p2[0])
    leg_labels.append('simulated')
    ax.set_xlabel('Years')
    ax.set_ylabel('Number of households')
    ax.set_ylim(ymin=0)
    leg = ax.legend(leg_lines, leg_labels)
    leg.get_frame().set_alpha(0.0)


def plot_pop_size(ax, sizes):
    """
    Plot population size over time.
    """
    scale_factor = float(sizes[100]) / data_pop_sizes[0]
    scaled_sizes = np.array(data_pop_sizes) * scale_factor
    p1 = ax.plot(data_years, scaled_sizes, color='b', 
            marker='o', markerfacecolor='b')
    p2 = ax.plot(range(1910, 1910+len(sizes[100:])), sizes[100:], color='g')
    leg_lines = []
    leg_labels = []
    leg_lines.append(p1[0])
    leg_labels.append('empirical')
    leg_lines.append(p2[0])
    leg_labels.append('simulated')
    ax.set_xlabel('Years')
    ax.set_ylabel('Population size')
    ax.set_ylim(ymin=0)
    leg = ax.legend(leg_lines, leg_labels)
    leg.get_frame().set_alpha(0.0)



def plot_hh_count_multi(ax, data):

    for d in data:
        scale_factor = 100.0 / d[0]
        scaled_counts = np.array(d) * scale_factor
        ax.plot(range(1910, 1910+len(d)), scaled_counts, color='0.4')
    scale_factor = 100.0 / data_counts[0]
    scaled_counts = np.array(data_counts) * scale_factor

    ax.plot(data_years, scaled_counts, color='k', marker='o', markerfacecolor='k')
    ax.set_xlabel('Years')
    ax.set_ylabel('Number of households')
    ax.set_ylim(ymin=0)



def plot_hh_size_time(ax, hh_sizes, comp=False):
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] 
    for i, cur_size in enumerate(zip(*hh_sizes)):
        x = range(1910, 1910+len(cur_size)) if comp else range(len(cur_size))
        ax.plot(x, cur_size, label='%d'%(i+1) if i<5 else '6+', lw=2, color=colors[i])
    x2 = [70,90]
    if comp:
        for i, zz in enumerate(comp_props):
            ax.plot(comp_years, np.array(zz)/100.0, marker='o', 
                    #mec=colors[i], 
                    mfc=colors[i], lw=1, color=colors[i], ls='--')
    ax.set_ylim((0.0, 0.5))
    leg = ax.legend(title='Household size', loc=9, ncol=2)
    leg.get_frame().set_alpha(0.0)
    if comp:
        ax.set_xlim((1910,2010))
    ax.set_xlabel('Years')
    ax.set_ylabel('Fraction of households')



def plot_fam_type_time(ax, fam_types, comp=False):
    lab = {'couple_kids': 'Couple with children', 'couple_only': 'Couple only', 'single_kids': 'Single with children'}
    colors = ['b', 'g', 'r', 'c', 'm', 'y'] 
    for i, k in enumerate(fam_types[0].keys()):
        cur_dat = [z[k] for z in fam_types]
        x = range(1910, 1910+len(cur_dat)) if comp else range(len(cur_dat))
        ax.plot(x, cur_dat, lw=2, label=lab[k], color=colors[i])
    leg = ax.legend(loc=6)
    leg.get_frame().set_alpha(0.0)
    if comp:
        ax.set_xlim((1910,2010))
    ax.set_xlabel('Years')
    ax.set_ylabel('Fraction of family households')


def plot_hh_lifecycle(ax, sizes, binsize=5):
    """
    Plot distribution of household sizes (using boxplot) by age of household.
    """

    # Transpose list of age-sequences to get a list of household sizes by age, 
    # removing null values.
    resizes = map(lambda *row: [e for e in row if e is not None], *sizes)

    # bin resizes into five year intervals
    #print resizes
    binned = []
    cur_bin = []
    for i, x in enumerate(resizes):
        cur_bin.extend(x)
        if i%binsize == 0:
            binned.append(cur_bin)
            cur_bin = []

    r = ax.boxplot(binned, positions=range(0, len(resizes), binsize), widths=3)
    for k in r.keys():
        plt.setp(r[k], color='black')
    plt.setp(r['medians'], color='red')
    ax.plot(range(len(resizes)), [np.mean(x) for x in resizes], color='k', lw=2)
    ax.set_xlabel('Household age')
    ax.set_ylim(ymin=0)
    ax.set_ylabel('Household size')


def build_legend_labels(cutoffs):
    """
    Create a list of series labels based upon cutoffs as follows:
    ['<c[0]', 'c[0]--c[1]-1', ..., 'c[n-1]--c[n]-1', 'c[n]+']
    """

    labels = ['<%d' % cutoffs[0]]
    labels += ['%d-%d'%(x,y-1) for x,y in zip(cutoffs[:-1],cutoffs[1:])]
    labels += ['%d+' % cutoffs[-1]]
    return labels


def plot_hh_comp(ax, counts, cutoffs, colours, binsize=1):
    """
    Plot household composition statistics (age breakdown) by, e.g., 
    household size or age.
    """

#    print counts

    x = np.arange(0, len(counts)*binsize, binsize)
    plots = []
    b = np.zeros((len(counts)), dtype=np.float)
    for i in range(len(cutoffs)+1):
        plots.append(ax.bar(x, counts[:,i], color=colours[i], bottom=b, 
            align='center', width=binsize*0.8))
        b += counts[:,i]
    ax.xaxis.set_ticks_position('none')
#    ax.set_xticklabels(np.arange(0, xl[1]*binsize, binsize, dtype=np.int))
    leg = ax.legend([p[0] for p in plots], build_legend_labels(cutoffs))
#    leg.get_frame().set_alpha(0.0)


def plot_hh_comp_by_hh_age(ax, data, cutoffs, colours, binsize):
    """
    Plot household household composition by household age.
    """

    plot_hh_comp(ax, data, cutoffs, colours, binsize)
    xl = 'Household age'
    if binsize>1:
        xl += ' (%d year bins)' % binsize
    ax.set_xlabel(xl)
    ax.set_ylabel('Fraction of individuals')
    xl = ax.get_xlim()
    ax.set_xlim((-0.4*binsize,xl[1]-1.6*binsize))
    ax.set_ylim((0.0, 1.0))
#    ax.set_title('Household composition by household age')
        

def plot_hh_comp_by_hh_size(ax, data, cutoffs, colours):
    """
    Plot household household composition by household size.
    """

    plot_hh_comp(ax, data, cutoffs, colours)
    ax.set_xlabel('Household size')
    ax.set_ylabel('Fraction of individuals')
    xl = ax.get_xlim()
    ax.set_xlim((0.6,xl[1]-0.6))
    ax.set_ylim((0.0, 1.0))
#    ax.set_title('Household composition by household size')
        

def plot_hh_comp_fig(hh_cc, colours):
    """
    Plot fancy dots household composition figure.

    NB: unlike other plotting functions, this one creates its own figure
    object as it needs control of the size.
    """

    # setup base figure
    xmax = max([x for x in hh_cc.keys()])+1
    ymax = max([len(x) for x in hh_cc.values()])+1
    fig = plt.figure(0,(xmax*.7,ymax*.7))
    ax = fig.add_subplot(111)
    #ax.set_title('Household composition by size and frequency')
    ax.set_xlabel('Household size')
    ax.set_ylabel('Frequency')
    #ax.xaxis.set_label_position('top')
    ax.xaxis.set_label_coords(0.15, 1.02)
    ax.yaxis.set_label_coords(-0.02, 0.9)
    ax.arrow(0.6, ymax-0.2, xmax/3, 0.0, lw=1, fc='k', 
            head_width=0.1, head_length=0.3)
    ax.arrow(0.6, ymax-0.2, 0.0, -ymax/4.5, lw=1, fc='k', 
            head_width=0.1, head_length=0.3)
    ax.set_xlim((0.5,xmax-0.5))
    ax.set_ylim((0,ymax))
    ax.set_xticks([])
    ax.set_yticks([])
    plt.setp(ax.xaxis.get_ticklines(), visible=False)
    for loc in ['left', 'right', 'bottom', 'top']:
        ax.spines[loc].set_color('none')
    inv = fig.transFigure.inverted()

    # for each (size, (hh_comp, count)) pair
    scale = 0.7
    for x,v in hh_cc.items():
        # for each (rank, hh_comp) pair, where rank is based on count
        for i, shh in enumerate(sorted(v, key=lambda x: x[1], reverse=True)):
            # invert y axis
            y = ymax-(i+1)
            # get location in terms of data
            tloc = inv.transform(ax.transData.transform((x, y)))
            # get scale adjustment
            wh_disp = ax.transData.transform((1, 1))
            # get offset adjustment
            zero_disp = ax.transData.transform((0, 0))
            # compute data to figure conversion
            wh = inv.transform((wh_disp[0]-zero_disp[0], 
                wh_disp[1]-zero_disp[1]))
            # create subplot for this hh comp
            a = plt.axes(
                    [tloc[0]-(scale*(wh[0]/2.)), tloc[1], 
                        wh[0]*scale, wh[1]*scale],
                    frame_on=False)
            # plot icon for this hh comp
            plot_hh_comp_icon_alt(a, shh[0], shh[1], colours)
            # add number of occurrences
            ax.text(x-0.3,y-0.19,'%d'%shh[1],size='xx-small')

    # create dummy objects for legend
    p = [Circle((0,0),5,edgecolor='none',facecolor=x) for x in colours]
    leg = ax.legend(p,('<5','5-17','18-64','65+'), loc=4)
    leg.get_frame().set_alpha(0.0)
    return fig


def plot_hh_comp_icon_alt(ax, hh_comp, freq, colours, a=1.0):
    """ 
    Helper function for plot_hh_comp_fig.  Plots an individual household icon.
    """

    pts = [[] for x in hh_comp]
    running = 0
    wh = int(ceil(sqrt(sum(hh_comp))))
    for c,i in enumerate(reversed(hh_comp)):
        for j in range(running, running+i):
            pts[c].append((j/wh,j%wh))
        running += i
    size = log(freq+1) * 20
    if len(pts[0])>0:
        ax.scatter(zip(*pts[0])[0], zip(*pts[0])[1], s=size, 
                linewidths=0,facecolor=colours[3], alpha=a)
    if len(pts[1])>0:
        ax.scatter(zip(*pts[1])[0], zip(*pts[1])[1], s=size, 
                linewidths=0,facecolor=colours[2], alpha=a)
    if len(pts[2])>0:
        ax.scatter(zip(*pts[2])[0], zip(*pts[2])[1], s=size, 
                linewidths=0,facecolor=colours[1], alpha=a)
    if len(pts[3])>0:
        ax.scatter(zip(*pts[3])[0], zip(*pts[3])[1], s=size, 
                linewidths=0,facecolor=colours[0], alpha=a)
#    ax.set_xlim(-.5,3.5)
#    ax.set_ylim(-.5,3.5)
    ax.set_xlim(-.5,wh-.5)
    ax.set_ylim(-.5,wh-.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


class d_plotter:
    """
    A class for plotting individual life event plots -- uses hack to allow
    pretty plotting of transparent filled curves.
    """

    def __init__(self):
        from matplotlib.patches import Rectangle
        self.c = {}
        self.c['red'] = Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.2) 
        self.c['green'] = Rectangle((0, 0), 1, 1, facecolor='green', alpha=0.2)
        self.c['blue'] = Rectangle((0, 0), 1, 1, facecolor='blue', alpha=0.2)
                
    def plot_age_dists(self, x, data, colours, labels, title, ofile, 
            show_legend=True):
        plt.clf()
        for d,c,l in zip(data, colours, labels):
            plt.fill_between(x, 0, d, facecolor=c, alpha=0.2, label=l)
        plt.xlabel('age')
        plt.ylabel('occurrences')
        if title:
            plt.title(title)
        if show_legend:
            leg = plt.legend([self.c[x] for x in colours], labels)
            leg.get_frame().set_alpha(0.0)
        plt.savefig(ofile)
        








