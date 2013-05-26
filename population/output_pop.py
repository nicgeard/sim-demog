import os, time, datetime
from glob import glob
from utils import load_probs, create_thumbnail
from data_processing_pop import *
from plotting_pop import *
from Cheetah.Template import Template

def output_html_report(sim, ofile='summary.html'):
    figs = []
    for infile in sorted(glob(os.path.join(sim.params['prefix'], '*.png'))):
        figs.append({
            'img': os.path.split(infile)[1], 
            'thumb': create_thumbnail(infile, (256,256)),
            'caption': ""})

    # create output file using template
    t = Template(file=os.path.join(sim.params['resource_prefix'], 
        'output.tmpl'), 
            searchList=[{
                'params': sim.params, 
                'images': figs, 
                'time': {
                    'begin': time.strftime("%a, %d %b %Y %H:%M:%S", 
                        time.localtime(sim.start_time)),
                    'end': time.strftime("%a, %d %b %Y %H:%M:%S", 
                        time.localtime(sim.end_time)),
                    'duration': datetime.timedelta(
                        seconds=sim.end_time-sim.start_time)}}])
    ofile = open(os.path.join(sim.params['prefix'], ofile), 'w')
    print >> ofile, t
    ofile.close()
    


def output_marriage_age_diffs(sim, ofile):
    d,b = np.histogram(sim.P.data['marriage_age_diffs'], \
            range(-10,10), normed=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(b[:-1], d, width=0.8, align='center')
    ax.set_xlabel('Age difference between male and female partners')
    ax.set_ylabel('Frequency')
    fig.savefig(ofile)

def output_birth_interval(sim, ofile):
    d,b = get_birth_interval_dist(sim.P.graveyard, 'not x.adam')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(b[:-1], d, width=0.8, align='center')
    ax.set_xlim(xmin=-0.5)
    ax.set_xlabel('Years between successive births')
    ax.set_ylabel('Frequency')
    fig.savefig(ofile)

def output_completed_fertility(sim, ofile):
    sample = [len(sim.P.I[x].children) \
        for x in sim.P.individuals_by_age(40,44) \
        if sim.P.I[x].sex == 1]
    d,b = np.histogram(sample, range(0,10), normed=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(b[:-1], d, width=0.8, align='center')
    ax.set_xlabel('Number of children ever born to women aged 40-44')
    ax.set_ylabel('Frequency')
    fig.savefig(ofile)


    #get_age_parity(sim.P.graveyard, 'not x.adam')

    ### INDIVIDUAL LIFE EVENT LOGS ######################################

def output_individual_logs(sim, ofile_prefix):
    marriage_1 = extract_distribution(sim.P.graveyard, 'not x.adam', 'm', 0)
    marriage_2 = extract_distribution(sim.P.graveyard, 'not x.adam', 'm', 1)
    marriage_3 = extract_distribution(sim.P.graveyard, 'not x.adam', 'm', 2)
    orphan = extract_distribution(sim.P.graveyard, 'not x.adam', 'gd')
    death = extract_distribution(sim.P.graveyard, 'not x.adam', 'd')
    divorce = extract_distribution(sim.P.graveyard, 'not x.adam', 's')
    c1f = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 0)
    c1 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 0, (15,51))
    c2 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 1, (15,51))
    c3 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 2, (15,51))
    c4 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 3, (15,51))
    c5 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 4, (15,51))
    c6 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 5, (15,51))
    c7 = extract_distribution(sim.P.graveyard, 'not x.adam', 'c', 6, (15,51))
    x = np.arange(0,100)

    cx = np.arange(15,50)

    dp = d_plotter()

    dp.plot_age_dists(x, (marriage_1[0], c1f[0], death[0]), \
            ('red', 'green', 'blue'), \
            ('first union', 'first child', 'death'), \
            #'major life events', 
            None, \
            os.path.join(ofile_prefix, 'major_life_events.%s'%sim.params['ext']))

    dp.plot_age_dists(cx, (c1[0], c2[0], c3[0], c4[0], c5[0]), \
            ('blue',)*5, \
            ('first', 'second', 'third', 'fourth', 'fifth'), \
            #'birth of children', 
            None, \
            os.path.join(ofile_prefix, 'birth_of_children.%s'%sim.params['ext']),
            show_legend=False)

    dp.plot_age_dists(x, (marriage_1[0], marriage_2[0], marriage_3[0]), \
            ('red', 'green', 'blue'), \
            ('first', 'second', 'third'), \
            #'marriages', 
            None, \
            sim.params['prefix'] + 'marriagesi.%s'%sim.params['ext'])

    # age_distribution by year of simulation
    # include currently alive individuals as well
    data_pop = dict(sim.P.graveyard, **sim.P.I)
    # this produces a list of [year, [births], [ages]] for each valid year
    # (current example is distribution of first births)
#    c1a = extract_annual_distributions(
#            data_pop, 'not x.adam', 'c', 0, (15, 51))
#    for x in c1a:
#        print x[0], x[1]


    ### HOUSEHOLD LIFE EVENT LOGS #######################################

# condition: doesn't exist at beginning or end of simulation
conditions = 'not x.adam and x.log[-1]["size"] == 0'
cutoffs = (5, 18, 65)
# must have one more colour than cutoff
#colours = ('#e86850', '#587498', '#587058', '#ffd800')
#colours = ('#990033', '#669900', '#0099FF', '#0033CC')
colours = ('r', 'c', 'b', 'y')

def output_hh_life_cycle(sim, ofile):
    """
    household life cycle diagram -- doesn't require data output?
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hh_sizes = size_by_age(sim.P.households, conditions, sim.params['years'])
    plot_hh_lifecycle(ax, hh_sizes)
    fig.savefig(ofile)

def output_hh_age_distribution(sim, ofile):
    """
    household age distribution -- not important
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    d,b = get_hh_age_dist(sim.P.households, conditions)
    ax.bar(b[1:-1],d[1:])
    ax.set_xlabel('Household age')
    ax.set_ylabel('Occurrences')
    fig.savefig(ofile)

def output_comp_by_hh_size(sim, ofile):
    """
    composition by household size -- to be integrated with dot plot
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_comp_by_hh_size(ax, 
            convert_counts_to_proportions(
                get_hh_comp_by_size(sim.P, cutoffs)), 
            cutoffs, colours)
    fig.savefig(ofile)

def output_comp_by_hh_age(sim, ofile):
    """
    composition by household age -- not particularly important
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_comp_by_hh_age(ax, 
            convert_counts_to_proportions(
                get_hh_comp_by_age(sim.P, cutoffs, 
                    sim.params['years'], binsize=5)),
            cutoffs, colours, 5)
    fig.savefig(ofile)

    ### GENERAL HOUSEHOLD AND POPULATION STATS ##########################

     
def output_hh_size_distribution(sim, max_size, ofile):
    """
    plot household size distribution
    """
    num_samples = int(len(sim.hh_size_dist) * 0.2)
    hh_means, hh_stdev = get_mean_and_stdev(sim.hh_size_dist[-num_samples:], max_size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if sim.params.has_key('hh_composition_final'):
        hh_dat = collapse_hh_size_probs(load_probs(os.path.join(
            sim.params['resource_prefix'], sim.params['hh_composition_final'])))
    else:
        hh_dat = collapse_hh_size_probs(load_probs(os.path.join(
            sim.params['resource_prefix'], sim.params['hh_composition'])))
    plot_hh_size_dist(ax, hh_means, hh_stdev, max_size, hh_dat)
    fig.savefig(ofile)

    outd = open(ofile+'.dat', 'w')
    outd.write(' '.join([str(x) for x in hh_means]))
    outd.write('\n')
    outd.write(' '.join([str(x) for x in hh_stdev]))
    outd.write('\n')
    outd.close()


def output_hh_size_multi(sim, dists, max_size, ofile):
    """
    plot household size distribution, using multiple runs.
    """
    hh_means, hh_stdev = get_mean_and_stdev(dists, max_size)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    num_samples = len(dists)
    if sim.params.has_key('hh_composition_final'):
        hh_dat = collapse_hh_size_probs(load_probs(os.path.join(
            sim.params['resource_prefix'], sim.params['hh_composition_final'])))
    else:
        hh_dat = collapse_hh_size_probs(load_probs(os.path.join(
            sim.params['resource_prefix'], sim.params['hh_composition'])))
    plot_hh_size_dist(ax, hh_means, hh_stdev, max_size, hh_dat)
    fig.savefig(ofile)


def output_age_distribution(sim, ofile):
    """
    plot age distribution
    -
    should also be dumpable as text, and aggregatable
    """
    num_samples = int(len(sim.age_dist) * 0.2)
    n_means, n_stdev = get_mean_and_stdev(sim.age_dist[-num_samples:])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if sim.params.has_key('age_distribution_final'):
        age_dat = np.array([x[0] for x in \
                load_probs(os.path.join(
            sim.params['resource_prefix'], 
            sim.params['age_distribution_final']))])
    else:
        age_dat = np.array([x[0] for x in \
                load_probs(os.path.join(
            sim.params['resource_prefix'], sim.params['age_distribution']))])
    plot_age_dist(ax, n_means, n_stdev, age_dat)
    fig.savefig(ofile)


def output_age_multi(sim, dists, ofile):
    # plot mean and stdev of age dists for multiple runs
    n_means, n_stdev = get_mean_and_stdev(dists)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if sim.params.has_key('age_distribution_final'):
        age_dat = np.array([x[0] for x in \
                load_probs(os.path.join(
            sim.params['resource_prefix'], 
            sim.params['age_distribution_final']))])
    else:
        age_dat = np.array([x[0] for x in \
                load_probs(os.path.join(
            sim.params['resource_prefix'], sim.params['age_distribution']))])
    plot_age_dist(ax, n_means, n_stdev, age_dat)
    fig.savefig(ofile)


def output_household_type(sim, ofile):
    # plot household type distributions 
    hh_stats = process_hh_stats(sim.P)
    fig = plot_hh_type(hh_stats)
    fig.savefig(ofile)


def output_household_type_multi(data, ofile):
    # plot mean and stdev of household type dists for multiple runs
    hh_mean, hh_stdev = process_hh_stats_multi(data)
    fig = plot_hh_type(hh_mean, hh_stdev)
    fig.savefig(ofile)


def output_household_composition(sim, ofile):
    # plot household compositions
    hh_cc = get_hh_comp_dict(sim.P)
    fig = plot_hh_comp_fig(hh_cc, colours)
    fig.savefig(ofile)


def output_hh_size_avg(sim, ofile):
    # plot average household size over time (most interesting for transition)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_size_avg(ax, sim.hh_size_avg)
    fig.savefig(ofile)
    outd = open(ofile+'.dat', 'w')
    outd.write('\n'.join([str(x) for x in sim.hh_size_avg]))
    outd.write('\n')
    outd.close()


def output_hh_count(sim, ofile):
    # plot average household count over time (most interesting for transition)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_count(ax, sim.hh_count)
    fig.savefig(ofile)
 

def output_pop_size(sim, ofile):
    # plot population size over time (most interesting for transition)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_pop_size(ax, sim.pop_size)
    fig.savefig(ofile)
 

def output_hh_trans_matrix(sim, ofile):
    np.savetxt(ofile, compare_hh_type(sim.P, 5), fmt='%.5f')


def output_hh_rates(sim, ofile):
    hh_birth_rates = np.average(sim.hh_rates['birth'][100:], 0)
    hh_death_rates = np.average(sim.hh_rates['death'][100:], 0)
    hh_in_rates = np.average(sim.hh_rates['hh_in'][100:], 0)
    hh_out_rates = np.average(sim.hh_rates['hh_out'][100:], 0)
    hh_size_dist = np.average(sim.hh_size_dist[100:], 0)

    hh_rates_file = open(ofile, 'wt')
    hh_rates_file.write('hh_size,'+
            ','.join([str(x) for x in range(1,len(hh_birth_rates)+1)])+'\n')
    hh_rates_file.write('dist,'+','.join([str(x) for x in hh_size_dist[:-1]])+'\n')
    hh_rates_file.write('birth,'+','.join([str(x) for x in hh_birth_rates])+'\n')
    hh_rates_file.write('death,'+','.join([str(x) for x in hh_death_rates])+'\n')
    hh_rates_file.write('hh_in,'+','.join([str(x) for x in hh_in_rates])+'\n')
    hh_rates_file.write('hh_out,'+','.join([str(x) for x in hh_out_rates])+'\n')
    hh_rates_file.close()


def output_hh_size_time(sim, ofile, comp):
    hh_sizes = collapse_large_households(sim.hh_size_dist[100:], 5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_size_time(ax, hh_sizes, comp)
    fig.savefig(ofile)


def output_hh_size_time_final100(sim, ofile, comp):
    hh_sizes = collapse_large_households(sim.hh_size_dist[:100], 5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_size_time(ax, hh_sizes, comp)
    fig.savefig(ofile)



def output_fam_type_time(sim, ofile, comp):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_fam_type_time(ax, sim.fam_types[100:], comp)
    fig.savefig(ofile)


from matplotlib.colors import LogNorm

def output_hh_contact_matrix(sim, ofile, cutoffs=None, age_scale=False, vmax=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if cutoffs:
        cm_data = get_hh_contact_matrix_cutoff(sim.P, cutoffs, age_scale)
        print cutoffs
        X, Y = np.meshgrid(np.array([0]+cutoffs), np.array([0]+cutoffs))
        print X
        print Y
        print cm_data
        if vmax:
            img = ax.pcolor(X, Y, cm_data, vmin=0.0, vmax=vmax,cmap='jet')
        else:
            img = ax.pcolor(X, Y, cm_data, vmin=0.0, cmap='jet')
    else:
        cm_data = get_hh_contact_matrix(sim.P, age_scale)
        img = ax.pcolor(cm_data, vmin=0.0, cmap='jet')    


    ax.set_aspect(1)
    ax.set_xlabel('Age')
    ax.set_ylabel('Age')
    fig.colorbar(img)
    fig.savefig(ofile+'.png')
    np.savetxt(ofile+'.csv', cm_data, fmt='%g', delimiter=',')

