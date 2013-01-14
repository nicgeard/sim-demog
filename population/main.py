"""
A stub for testing/evaluating dynamic demography simulations.
"""

import cPickle as pickle
import sys, os 
from utils.utils import parse_params, create_path
from data_processing_pop import *
from plotting_pop import *
from output_pop import *
from simulation import Simulation
import numpy as np
from random import Random


def load_params():
    print os.path.dirname(__file__)
    params = parse_params('params.cfg', os.path.join(
        os.path.dirname(__file__), 'paramspec_pop.cfg'))
    create_path(params['prefix'])
    create_path(os.path.join(params['prefix'], 'thumbnails')) 
    print params
    return params


def run_single(params):
    print "Creating population..."
    sim = Simulation(params)
    timesteps = params['years'] * (365/params['t_dur'])
    print "Running simulation..."
    sim.start_time = time.time()
    for i in xrange(timesteps):
        t = i * params['t_dur']
        b,d,im,bd = sim.update_all_demo(t)
        print i, t, len(sim.P.I), len(b), len(d), len(im)
        sim.record_stats_demo(t)
#        if t>36500: # exclude first 100 years
#            for ind in sim.P.I.values():
#                ind.hh_type.append(sim.P.get_hh_type(ind))
    sim.end_time = time.time()
    return sim


def save_sim(sim):
    print "Saving simulation..."
    sim.P.flatten()
    pickle.dump(sim,
            open(os.path.join(sim.params['prefix'], 'sim.p'),'wb'),
            pickle.HIGHEST_PROTOCOL)
    pickle.dump(sim.P,
            open(os.path.join(sim.params['prefix'], 'burnt.pop'),'wb'),
            pickle.HIGHEST_PROTOCOL)
    sim.P.lift()


def load_sim(p):
    print "Loading simulation..."
    sim = pickle.load(open(os.path.join(p['prefix'], 'sim.p'),'rb'))
    sim.P.lift()
    return sim


def go_single(p):
    if p['random_seed']:
        p['seed'] = Random().randint(0, 99999999)
    sim = run_single(p)
    save_sim(sim)
    return sim


def go_multi(p):
    seed_rng = Random() if p['random_seed'] else Random(p['seed']) 

    fig_prefix = p['prefix']

    age_file = file(os.path.join(fig_prefix, 'age_multi.dat'), 'w')
    hh_size_file = file(os.path.join(fig_prefix, 'hh_size_multi.dat'), 'w')
    pickle.dump([], file(os.path.join(fig_prefix, 'hh_type_multi.dat'), 'wb'))
    avg_hh_size_file = file(os.path.join(fig_prefix, 'avg_hh_multi.dat'), 'w')
    hh_count_file = file(os.path.join(fig_prefix, 'hh_count_multi.dat'), 'w')
    hh_change_file = file(os.path.join(fig_prefix, 'hh_change_multi.dat'), 'w')
    fam_change_file = file(os.path.join(fig_prefix, 'fam_change_multi.dat'), 'w')

    for z in range(p['num_runs']):
        print "------", z, "------"
        p['seed'] = seed_rng.randint(0, 99999999)
        sim = run_single(p)

        age_file.write('%d ' % sim.params['seed'])
        age_file.write(' '.join([str(x) for x in sim.P.age_dist(101,101)[0]]))
        age_file.write('\n')
        age_file.flush()

        hh_size_file.write('%d ' % sim.params['seed'])
        hh_size_file.write(' '.join([str(x) for x in \
                sim.P.group_size_dist('household', 10)[0]]))
        hh_size_file.write('\n')
        hh_size_file.flush()

        hh_change_file.write('%d ' % sim.params['seed'])
        dist_a = np.array(collapse_large_households([sim.hh_size_dist[-40]], 6)[0])
        dist_z = np.array(collapse_large_households([sim.hh_size_dist[-20]], 6)[0])
        change = (dist_z - dist_a) / dist_a
        hh_change_file.write(' '.join([str(x) for x in change])) 
        hh_change_file.write('\n')
        hh_change_file.flush()

        fam_change_file.write('%d ' % sim.params['seed'])
        dist_a = np.array([sim.fam_types[-21]['couple_kids']] + [sim.fam_types[-21]['couple_only']])
        dist_z = np.array([sim.fam_types[-1]['couple_kids']] + [sim.fam_types[-1]['couple_only']])
        change = (dist_z - dist_a) / dist_a
        fam_change_file.write(' '.join([str(x) for x in change])) 
        fam_change_file.write('\n')
        fam_change_file.flush()

        cur_stats = pickle.load(open(os.path.join(fig_prefix, 'hh_type_multi.dat'), 'rb'))
        cur_stats.append(process_hh_stats(sim.P))
        pickle.dump(cur_stats, open(os.path.join(fig_prefix, 'hh_type_multi.dat'), 'wb'))

        avg_hh_size_file.write('%d ' % sim.params['seed'])
        avg_hh_size_file.write(' '.join([str(x) for x in sim.hh_size_avg]))
        avg_hh_size_file.write('\n')
        avg_hh_size_file.flush()

        hh_count_file.write('%d ' % sim.params['seed'])
        hh_count_file.write(' '.join([str(x) for x in sim.hh_count]))
        hh_count_file.write('\n')
        hh_count_file.flush()

    age_file.close()
    hh_size_file.close()
    avg_hh_size_file.close()
    hh_count_file.close()
    hh_change_file.close()
    fam_change_file.close()

####
#### NEED TO move hhsize multi and hhtype multi into "plot_multi" file...
#### ALSO mean and stdev trans calculations (should be written into file?)
####

def output_multi(p):
    """
    Output data aggregated across multiple simulations.  This is a bit ugly, 
    as it involves loading data from files, rather than transferring across
    directly.  However, this makes for better recovery from interrupted runs.
    """

    max_hh_size = 6
#    max_hh_size = 10
    fig_prefix = p['prefix']
    create_path(fig_prefix)

    "Age structure"
    data_file = file(os.path.join(fig_prefix, 'age_multi.dat'), 'r')
    age_data = []
    for line in data_file:
        a = line.split()
        age_data.append([float(x) for x in a[1:]])
    age_real = np.array([x[0] for x in load_probs(
#        '/home/ngeard/projects/demography/data/age_structure_zambia_2000.dat')])
        '/home/ngeard/projects/demography/data/age_dist.dat')])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_age_dist_multi(ax, age_data, age_real)
    fig.savefig('output/new_age_dists_multi.%s'%p['ext'])

    "Household size distribution"
    data_file = file(os.path.join(fig_prefix, 'hh_size_multi.dat'), 'r')
    hh_size_data = []
    for line in data_file:
        a = line.split()
        hh_size_data.append([float(x) for x in a[1:]])
    print hh_size_data
    hh_size_means, hh_size_stdev = get_mean_and_stdev(hh_size_data, max_hh_size)
    hh_size_real = np.array([x[0] for x in load_probs(
#        '/home/ngeard/projects/demography/data/hh_size_dist_zambia.dat')])
        '/home/ngeard/projects/demography/data/hh_size_dist.dat')])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_size_dist(ax, hh_size_means, hh_size_stdev, max_hh_size, hh_size_real, 'initial')
    fig.savefig(os.path.join(fig_prefix, 'new_hh_size_dists_multi.%s'%p['ext']))

    "Household type distribution"
    hh_type_data = pickle.load(file(os.path.join(fig_prefix, 'hh_type_multi.dat'), 'rb'))
    print hh_type_data
    hh_type_mean, hh_type_stdev = process_hh_stats_multi(hh_type_data)

    fig = plot_hh_type(hh_type_mean, hh_type_stdev)
    fig.savefig('output/new_hh_type.%s'%p['ext'])

    "Household change distribution"
    hh_change_file = file(os.path.join(fig_prefix, 'hh_change_multi.dat'), 'r')
    hh_change_data = []
    for line in hh_change_file:

        a = line.split()
        hh_change_data.append([float(x) for x in a[1:]])
    print hh_change_data
    hh_change_means, hh_change_stdev = get_mean_and_stdev(hh_change_data, 6)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_change_dist(ax, hh_change_means, hh_change_stdev, 6)
    fig.savefig(os.path.join(fig_prefix, 'hh_change_multi.%s'%p['ext']))


    "Average household size over time"
    data_file = file(os.path.join(fig_prefix, 'avg_hh_multi.dat'), 'r')
    hh_data = []
    for line in data_file:
        a = line.split()
        hh_data.append([float(x) for x in a[101:]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_avg_hh_multi(ax, hh_data)
    fig.savefig('output/avg_hh_multi.%s'%p['ext'])

    "Number of households over time"
    data_file = file(os.path.join(fig_prefix, 'hh_count_multi.dat'), 'r')
    hh_data = []
    for line in data_file:
        a = line.split()
        hh_data.append([float(x) for x in a[101:]])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_hh_count_multi(ax, hh_data)
    fig.savefig('output/hh_count_multi.%s'%p['ext'])
    

def output_single(sim):
    print "Processing data and generating output plots..."

    fig_prefix = sim.params['prefix']
    create_path(fig_prefix)

#    output_marriage_age_diffs(sim, os.path.join(fig_prefix, 'marriage_age_diffs.png'))
#    output_birth_interval(sim, os.path.join(fig_prefix, 'birth_interval.%s'%sim.params['ext']))
#    output_individual_logs(sim, fig_prefix)
#    output_hh_life_cycle(sim, os.path.join(fig_prefix, 'hh_life_cycle.%s'%sim.params['ext']))
#    output_hh_age_distribution(sim, os.path.join(fig_prefix, 'hh_age_dist.%s'%sim.params['ext']))
#    output_comp_by_hh_size(sim, os.path.join(fig_prefix, 'age_by_hh_size.%s'%sim.params['ext']))
#    output_comp_by_hh_age(sim, os.path.join(fig_prefix, 'age_by_hh_age.%s'%sim.params['ext']))   
#    output_hh_size_distribution(sim, 10, os.path.join(fig_prefix,'hh_size_dist.%s'%sim.params['ext']))
#    output_age_distribution(sim, os.path.join(fig_prefix,'age_dist.%s'%sim.params['ext']))
#    output_household_type(sim, os.path.join(fig_prefix,'hh_type.%s'%sim.params['ext']))
#    output_household_composition(sim, os.path.join(fig_prefix,'hh_comp.%s'%sim.params['ext']))
#    output_completed_fertility(sim, os.path.join(fig_prefix,'completed_fertility.%s'%sim.params['ext']))
#    output_hh_size_avg(sim, os.path.join(fig_prefix,'hh_size_avg.%s'%sim.params['ext']))
#    output_hh_count(sim, os.path.join(fig_prefix,'hh_count.%s'%sim.params['ext']))
#    output_pop_size(sim, os.path.join(fig_prefix,'pop_size.%s'%sim.params['ext']))
#    output_hh_rates(sim, os.path.join(fig_prefix, 'hh_rates.csv'))
#    output_hh_trans_matrix(sim, os.path.join(fig_prefix, 'trans-%d'%sim.params['seed']))
#    output_hh_size_time(sim, os.path.join(fig_prefix, 'hh_size_time.%s'%sim.params['ext']), comp=False)
#    output_hh_size_time_final100(sim, os.path.join(fig_prefix, 'hh_size_time_final100.%s'%sim.params['ext']), comp=False)
#    output_fam_type_time(sim, os.path.join(fig_prefix, 'fam_type_time.%s'%sim.params['ext']), comp=False)
#    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_unscaled'))
#    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_binned'), [5,12,20,40,65,100])
#    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_binned'), [5*x for x in range(21)])
#    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_old_binned_scaled'), [x for x in range(1,80)]+[100], True)
    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_binned_scaled'), [5*x for x in range(1,16)]+[100], True)#, 0.005)
    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_trish'), [1,6,13,20,40,60,100], True, 0.005)
#    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_0'), [100], True)
    #output_html_report(sim)
#    output_hh_contact_matrix(sim, os.path.join(fig_prefix, 'cm_scaled'), None, True, vmax=1.0)

    #TODO: how to match captions? perhaps "output_all()" could return an 
    # appropriate figs list, with captions and subsections specified.

    print 'Output written to ', sim.params['prefix']

if __name__ == '__main__':
    p = load_params()
    p['ext'] = 'svg'
    if p['num_runs'] < 2:
        sim = go_single(p) if len(sys.argv) <= 1 else load_sim(p)
        if p['logging']: output_single(sim)
    else:
        if len(sys.argv) <= 1: go_multi(p)
        if p['logging']: output_multi(p)


