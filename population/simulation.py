from __future__ import absolute_import
import time, os
import numpy as np
from random import Random
from math import exp
from population.pop_hh import Pop_HH
from population.individual import Individual
from utils.utils import sample_table, adjust_prob, load_probs, load_probs_new, load_age_rates, load_prob_tables, load_prob_list


"""
simulation.py

Agent-based simulation a population with pseudo-realistic demography.
"""

def adjust_prob(P, t_dur):
    N = t_dur/365.0
    return 1 - pow(1-P, N)


class Simulation(object):

    def __init__(self, params, ind_type=Individual, create_pop=True):
        # convert ConfigObj to dictionary to store params
        self.params = dict(params)
        self.params_adj={}
        self.rng = Random(self.params['seed'])
        self.load_demographic_data()
        if create_pop:
            self.create_population(ind_type, params['logging'])

        # for storing general data
        self.max_hh = 50 # maximum possible hh size
        self.pop_size = []
        self.age_dist = []
        self.hh_size_dist = []
        self.hh_size_dist_counts = []
        self.hh_size_avg = []
        self.hh_count = []
        self.fam_types = []
        self.start_time = None
        self.end_time = None



    def create_population(self, ind_type, logging=True):
        """
        Create a population according to specified age and household size
        distributions.
        """

        self.P = Pop_HH(ind_type, logging)
        self.P.gen_hh_age_structured_pop(self.params['pop_size'], self.hh_comp, 
                self.age_dist, self.params['age_cutoffs'], self.rng)
        self.P.allocate_couples()
        self.P.print_population_summary()


    def parse_age_rates(self, filename, factor, final):
        """
        Parse an age-year-rate table to produce a dictionary, keyed by age, 
        with each entry being a list of annual rates (by year).
        
        Setting final to 'True' appends an age 100 rate of >1 (e.g., to 
        ensure everyone dies!
        """

        dat = load_age_rates(filename)
        rates = {}
        for line in dat:
            rates[line[0]] = [x * factor for x in line[1:]]
        if final:
            rates[101] = [100 for x in dat[0][1:]] # everybody dies...
        return rates


    def load_demographic_data(self):
        """
        Load data on age-specific demographic processes (mortality/fertility)
        and adjust event probabilities according to time-step.
        """

        # load household size distribution and age distribution
        self.hh_comp = load_probs(os.path.join(self.params['resource_prefix'], 
                    self.params['hh_composition']), False)
        self.params['age_cutoffs'] = [int(x) for x in self.hh_comp[0][1:][0]]  # yuk!
        self.age_dist = load_probs(os.path.join(self.params['resource_prefix'], 
                    self.params['age_distribution']))

        annual_factor = self.params['t_dur']/365.0

        # load and scale MORTALITY rates
        self.death_rates = {}
        self.death_rates[0] = self.parse_age_rates(os.path.join(
            self.params['resource_prefix'], 
            self.params['death_rates_m']), annual_factor, True)
        self.death_rates[1] = self.parse_age_rates(os.path.join(
            self.params['resource_prefix'], 
            self.params['death_rates_f']), annual_factor, True)

        ### load FERTILITY age probs (don't require scaling) for closed pops
        self.fertility_age_probs = load_prob_tables(os.path.join(
            self.params['resource_prefix'], 
            self.params['fertility_age_probs']))
        self.fertility_parity_probs = load_probs_new(os.path.join(
            self.params['resource_prefix'],
            self.params['fertility_parity_probs']))

        ### load and scale leav/couple/divorce and growth rates
        if self.params['dyn_rates']:
            # rates will be a list of annual values
            self.params['leaving_probs'] = load_prob_list(os.path.join(
                self.params['resource_prefix'], self.params['leaving_prob_file']))
            self.params['couple_probs'] = load_prob_list(os.path.join(
                self.params['resource_prefix'], self.params['couple_prob_file']))
            self.params['divorce_probs'] = load_prob_list(os.path.join(
                self.params['resource_prefix'], self.params['divorce_prob_file']))
            self.params['growth_rates'] = load_prob_list(os.path.join(
                self.params['resource_prefix'], self.params['growth_rate_file']))
            self.params['imm_rates'] = load_prob_list(os.path.join(
                self.params['resource_prefix'], self.params['imm_rate_file']))

            self.params_adj['leaving_probs'] = [adjust_prob(x, self.params['t_dur']) 
                for x in self.params['leaving_probs']]
            self.params_adj['couple_probs'] = [adjust_prob(x, self.params['t_dur']) 
                for x in self.params['couple_probs']]
            self.params_adj['divorce_probs'] = [adjust_prob(x, self.params['t_dur']) 
                for x in self.params['divorce_probs']]
            self.params_adj['growth_rates'] = [adjust_prob(x, self.params['t_dur']) 
                for x in self.params['growth_rates']]
            self.params_adj['imm_rates'] = [adjust_prob(x, self.params['t_dur']) 
                for x in self.params['imm_rates']]

            self.dyn_years = min(len(self.death_rates[0][0])-1, len(self.fertility_age_probs)-1,
                    len(self.params_adj['leaving_probs'])-1, len(self.params_adj['couple_probs'])-1,
                    len(self.params_adj['divorce_probs'])-1, len(self.params_adj['growth_rates'])-1)

        else:
            # adjust demographic event probabilities according to time step
            self.params_adj['couple_probs'] = [adjust_prob(
                self.params['couple_prob'], self.params['t_dur'])]
            self.params_adj['leaving_probs'] = [adjust_prob(
                self.params['leaving_prob'], self.params['t_dur'])]
            self.params_adj['divorce_probs'] = [adjust_prob(
                self.params['divorce_prob'], self.params['t_dur'])]
            self.params_adj['growth_rates'] = [adjust_prob(
                self.params['growth_rate'], self.params['t_dur'])]
            self.params_adj['imm_rates'] = [adjust_prob(
                self.params['imm_rate'], self.params['t_dur'])]


    def update_individual_demo(self, t, ind, index=0):
        """
        Update individual ind; check for death, couple formation, leaving home
        or divorce, as possible and appropriate.
        """

        death = None; birth = None

        couple_prob = self.params_adj['couple_probs'][index]
#        if ind.divorced and ind.deps: couple_prob *= 0.5

        # DEATH / BIRTH: 
        if self.rng.random() > exp(-self.death_rates[ind.sex][ind.age][index]):
            death = ind
            mother = ind
            while mother is ind:    # make sure dead individual isn't selected as mother!
                mother = self.choose_mother(index)
            birth = self.update_death_birth(t, ind, mother)

        # COUPLE FORMATION:
        elif self.params['couple_age'] < ind.age < 60 \
                and not ind.partner \
                and self.rng.random() < couple_prob:
            partner = self.choose_partner(ind)
            if partner:
                self.P.form_couple(t, ind, partner)

        # LEAVING HOME:
        elif ind.age > self.params['leaving_age'] \
                and ind.with_parents \
                and not ind.partner \
                and self.rng.random() < self.params_adj['leaving_probs'][index]:
            self.P.leave_home(t, ind)

        # DIVORCE:
        elif self.params['divorce_age'] < ind.age < 50 \
                and ind.partner \
                and self.rng.random() < self.params_adj['divorce_probs'][index]:
            self.P.separate_couple(t, ind)

        # ELSE: individual has a quiet year...
        return death, birth 


    def choose_mother(self, index):
        """
        Choose a new mother on the basis of fertility rates.

        NOTE: there is still a very small possibility (in *very* small
        populations) that this will hang due to candidates remaining
        forever empty.  Should add a check to prevent this and exit gracefully.
        """

        candidates = []
        while not candidates:
            tgt_age = int(sample_table(self.fertility_age_probs[index], self.rng)[0])
            tgt_prev_min = 0; tgt_prev_max = 100
            if self.params['use_parity']:
                tgt_prev_min = int(sample_table(
                    self.fertility_parity_probs[(tgt_age-15)/5], self.rng)[0])
                # effectively transform 5 into 5+
                tgt_prev_max = tgt_prev_min if tgt_prev_min < 5 else 20
            tgt_set = self.P.individuals_by_age(tgt_age, tgt_age)
            candidates = [x
                    for x in tgt_set \
                    if x.sex == 1 \
                    and x.can_birth() \
                    and not x.with_parents \
                    and tgt_prev_min <= len(x.children) <= tgt_prev_max
                    ]
        return self.rng.choice(candidates)


    def choose_partner(self, ind):
        """
        Choose a partner for i_id, subject to parameter constraints.

        :param ind: the first partner in the couple.
        :type ind: Individual
        :returns: partner if successful, otherwise None.
        """

        mean_age = ind.age+self.params['partner_age_diff'] \
                if ind.sex == 0 else ind.age-self.params['partner_age_diff']
        tgt_age = 0
        while tgt_age < self.params['min_partner_age']:
            tgt_age = int(self.rng.gauss(mean_age, self.params['partner_age_sd']))
            tgt_set = self.P.individuals_by_age(tgt_age, tgt_age)
            candidates = [x \
                for x in tgt_set \
                if not x.partner \
                and x.sex != ind.sex \
                and x not in self.P.hh_members(ind)
                ]

        # abort if no eligible partner exists
        return None if candidates == [] else self.rng.choice(candidates)



    def update_death_birth(self, t, ind, mother):
        """
        Replace a dying individual with a newborn.  If no individual to die
        is passed, only a birth occurs; if no mother is passed, only a death
        occurs.

        :param t: the current time step.
        :type t: int
        :param ind: the individual to die.
        :type ind: Individual
        :returns: the new individual.
        """

        if ind:
            orphans = self.P.death(t, ind)
            self.P.process_orphans(t, orphans, self.params['age_cutoffs'][-2], self.rng)
        
        if mother:
            sex = self.rng.randint(0, 1)
            new_ind = self.P.birth(t, self.rng, mother, mother.partner, sex)
            return new_ind
        
        return None



    def update_all_demo(self, t):
        """
        Update population over period of t days.

        Returns list of births, deaths, immigrants and birthdays.
        """

        birthdays = self.P.age_population(self.params['t_dur'])

        deaths = []; births = []

        # calculate index for fertility and mortality rates
        # basically: use first entry for burn-in, then one entry every 
        # 'period' years, then use the final entry for any remaining years.
        index = min(max(0, (t - (self.params['demo_burn']*365)) / 
                (self.params['demo_period']*365)), self.dyn_years) \
                        if self.params['dyn_rates'] else 0

        cur_inds = self.P.I.values()
        for ind in cur_inds: 
            death, birth = self.update_individual_demo(t, ind, index)
            if death: deaths.append(death)
            if birth: births.append(birth)

        #population growth
        for x in xrange(int(len(self.P.I) * self.params_adj['growth_rates'][index])):
            mother = self.choose_mother(index)
            births.append(self.update_death_birth(t, None, mother))

        #immigration
        imm_count = 0
        imm_tgt = int(len(self.P.I) * self.params_adj['imm_rates'][index])
        source_hh_ids = []
        immigrants = []
        while imm_count < imm_tgt:
            hh_id = self.rng.choice(self.P.groups['household'].keys())
            imm_count += len(self.P.groups['household'][hh_id])
            source_hh_ids.append(hh_id)
        for hh_id in source_hh_ids:
            new_hh_id = self.P.duplicate_household(t, hh_id)
            immigrants.extend(self.P.groups['household'][new_hh_id])
        
        return births, deaths, immigrants, birthdays



    def record_stats_demo(self, t):
        if self.params['record_interval'] <= 0 or t%self.params['record_interval'] is not 0:
            return
        
        self.pop_size.append(len(self.P.I))
        self.age_dist.append(self.P.age_dist(101,101)[0])
        self.hh_size_dist.append(self.P.group_size_dist('household', self.max_hh)[0])
        self.hh_size_dist_counts.append(self.P.group_size_dist('household',self.max_hh,False)[0])
        self.hh_size_avg.append(self.P.group_size_avg('household'))
        self.hh_count.append(len(self.P.groups['household']))

        self.fam_types.append(self.P.sum_hh_stats_group())



    def run(self):
        """
        Run simulated population (NB: this probably won't be used, as
        external running object may wish to just call update_all itself...)
        """

        timesteps = self.params['years'] * (365/self.params['t_dur'])
        i = 0
        while i < timesteps:
            t = i*self.params['t_dur']
            print i, t
            self.update_all_demo(t)
            i += 1
 



