"""
A stub for testing/evaluating dynamic demography simulations.
"""

import cPickle as pickle
import sys, os 
from utils import parse_params, create_path
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
    sim.end_time = time.time()
    return sim


def save_sim(sim, fname):
    print "Saving simulation..."
    sim.P.flatten()
    pickle.dump(sim, open(fname,'wb'), pickle.HIGHEST_PROTOCOL)
    sim.P.lift()


def load_sim(fname):
    print "Loading simulation..."
    sim = pickle.load(open(fname,'rb'))
    sim.P.lift()
    return sim


def go_single(p):
    if p['random_seed']:
        p['seed'] = Random().randint(0, 99999999)
    fname = os.path.join(p['prefix'], 'final_%d.sim'%p['seed'])
    if os.path.isfile(fname):
        sim = load_sim(fname)
    else:
        sim = run_single(p)
        save_sim(sim, fname)
    return sim


    

def output_single(sim):
    print "Processing data and generating output plots..."

    fig_prefix = sim.params['prefix']
    create_path(fig_prefix)

    output_hh_life_cycle(sim, os.path.join(fig_prefix, 'hh_life_cycle.%s'%sim.params['ext']))
    output_comp_by_hh_size(sim, os.path.join(fig_prefix, 'age_by_hh_size.%s'%sim.params['ext']))
    output_comp_by_hh_age(sim, os.path.join(fig_prefix, 'age_by_hh_age.%s'%sim.params['ext']))   
    output_hh_size_distribution(sim, 10, os.path.join(fig_prefix,'hh_size_dist.%s'%sim.params['ext']))
    output_age_distribution(sim, os.path.join(fig_prefix,'age_dist.%s'%sim.params['ext']))
    output_household_type(sim, os.path.join(fig_prefix,'hh_type.%s'%sim.params['ext']))
    output_household_composition(sim, os.path.join(fig_prefix,'hh_comp.%s'%sim.params['ext']))
    output_hh_size_time(sim, os.path.join(fig_prefix, 'hh_size_time.%s'%sim.params['ext']), comp=False)
    output_fam_type_time(sim, os.path.join(fig_prefix, 'fam_type_time.%s'%sim.params['ext']), comp=False)


    print 'Output written to ', sim.params['prefix']


if __name__ == '__main__':
    p = load_params()
    p['ext'] = 'svg'
    sim = go_single(p)
    output_single(sim)


