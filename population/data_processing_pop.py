import numpy as np
from collections import defaultdict

def extract_distribution(I, condition, code, rank=0, age_dist=(0,101)):
    """
    Retrieve the age distribution of a particular type of event from 
    individual life histories.

    :param I: the list of individuals to analyse.
    :type I: list
    :param condition: an evaluatable condition used to filter the list of individuals (e.g., by sex, etc.)
    :type condition: string
    :param code: the event code to collect the age distribution for.
    :type code: string
    :param rank: the rank of the event, beginning at 0 (e.g., use 2 to collect the 3rd birth).
    :type rank: int
    :param age_dist: age range to bin over.
    :type age_dist: tuple
    """
    age_data = []

    # for each individual in population where 'condition' is satisfied;
    for x in [x for x in I.values() if eval(condition)]:
        # search for appropriate event code; may want later entries if 
        # looking for subsequent events, eg. second child, marriage, etc.
        ages = [y['age'] for y in x.log if y['code'] == code]
        if len(ages) > rank:
            age_data.append(ages[rank])

    return np.histogram(age_data, 
            bins=range(age_dist[0]+1,age_dist[1]+1), normed=False)


def extract_annual_distributions(I, condition, code, rank=0, 
        age_dist=(0,101)):
    """
    As above, except that one distribution is generated for each 
    simulation year.

    NOTE: burn in years aren't automatically excluded (this is done 
    indirectly via the fact that bootstrap individuals won't be 
    included in 'I', so there will be many 'empty' distributions to begin.
    """
    age_data = defaultdict(list)

    # for each individual in population where 'condition' is satisfied;
    for x in [x for x in I.values() if eval(condition)]:
        age_years = [(y['age'], y['time']/365) \
                for y in x.log if y['code'] == code]
        #print age_years
        if len(age_years) > rank:
            age_data[age_years[rank][1]].append(age_years[rank][0])
    
    hists = []
    for year, year_data in age_data.items():
        hists.append([year]+list(np.histogram(year_data,
            bins=range(age_dist[0]+1,age_dist[1]+1), normed=False)))
    return hists


def get_birth_interval_dist(I, condition):
    """
    Retrieve the distribution of birth intervals in a population.

    :param I: the list of individuals to analyse.
    :type I: list
    :param condition: an evaluatable condition used to filter the list of individuals (e.g., by sex, etc.)
    :type condition: string
    """
    birth_intervals = []

    for x in [x for x in I.values() if eval(condition)]:
        ages = [y['age'] for y in x.log if y['code'] == 'c']
        diffs = [z-y for z, y in zip(ages[1:], ages[:-1])]
        birth_intervals.extend(diffs)

    return np.histogram(birth_intervals, range(20), normed=True)


def get_age_parity(I, condition, cutoffs=[20,25,30,35,40,45,50,55,60]):
    """
    Calculate the fraction of births to women with 0, 1, 2, 3, 4, or 5 or more previous children, broken down by age group.

    :param I: the list of individuals to analyse.
    :type I: list
    :param condition: an evaluatable condition used to filter the list of individuals (e.g., by sex, etc.)
    :type condition: string
    :param cutoffs: the age bracket cutoffs to use.
    :type cutoffs: list
    """
    counts = np.zeros((len(cutoffs)+1, 6))
    for x in [x for x in I.values() if eval(condition)]:
        ages = [y['age'] for y in x.log if y['code'] == 'c']
        for order,cur_age in enumerate(ages):
            if order>5: order=5
            i = 0
            done = False
            while not done:
                if cur_age < cutoffs[i]:
                    counts[i][order] += 1
                    done = True
                i += 1
                counts[-1][order] += 1

    sums = np.sum(counts,1,dtype=np.float32)
    sums = sums.reshape((len(sums),1))
#    print counts/sums


def collapse_hh_size_probs(probs):
    '''
    Collapse the hh_comp.dat file to remove age distinctions.

    NB: where is this used?
    '''
    c_probs = [0.0]*30 # 30 == max number of possible household types
    summed = [[x[0],sum([int(y) for y in x[1]])] for x in probs]
    for x in summed:
        c_probs[x[1]]+=x[0]
    return c_probs


def size_by_age(logs, condition, max_t):
    """
    Processes household logs to produce a list of household size vectors.
    Each element i of a vector contains the size of the household in year i.

    e.g., [2, 2, 2, 3, 3, 3, 4, 4, 4, ..., 3, 3, 3, 2, 2, 2, 1, 1, 0]

    :param logs: the list of households to analyse.
    :type logs: list
    :param condition: an evaluatable condition used to filter the list of households.
    :type condition: string
    :param max_t: the endpoint of the simulation.
    :type max_t: int
    """
    sizes = []

    for hh in [x for x in logs.values() if eval(condition)]:
        # final age equal to max_t - formation_time for extant households
        if hh.log[-1]['size'] > 0:  # final size is 0
            hh.log.append({'age':max_t-hh.log[0]['time'], 'size':'x'})
        else:   # house still exists
            hh.log.append({'age':0, 'size':'x'})
        cur_sizes = []
        [cur_sizes.extend([x['size']]*(y['age']/365-x['age']/365)) 
                for x, y in zip(hh.log[:-1], hh.log[1:])]
        sizes.append(cur_sizes)
        del hh.log[-1]

    return sizes


def get_hh_age_dist(logs, condition):
    """
    Retrieve the distribution of household ages (in years) from a set of household logs.

    :param logs: the list of households to analyse.
    :type logs: list
    :param condition: an evaluatable condition used to filter the list of households.
    :type condition: string
    """
    ages = []
    for hh in [x for x in logs.values() if eval(condition)]:
        ages.append(hh.log[-2]['age']/365)
    return np.histogram(ages, bins=range(101), normed=False)



def get_hh_comp_by_size(P, cutoffs):
    """
    Retrieve age composition of households by household size. NB: this is
    a snapshot of the population, rather than an aggregate over historical data.

    :params P: the population.
    :type P: :class:`pop_hh.Pop_HH`
    :params cutoffs: the list of cutoffs for age categories (e.g.,[5,18,65])
    :type cutoffs: list
    """
    cutoffs = list(cutoffs) + [200]
    counts = np.zeros((25, len(cutoffs)))

    for hh in P.groups['household'].values():
        cur_size = len(hh)
        for i_id in hh:
            cur_age = i_id.age
            i = 0
            while cur_age >= cutoffs[i]:
                i += 1
            counts[cur_size][i] += 1
    return counts


def get_hh_comp_by_age(P, cutoffs, cur_year, binsize=1):
    """
    Retrieve age composition of households by household age. NB: this is
    a snapshot of the population, rather than an aggregate over historical data.

    :params P: the population.
    :type P: :class:`pop_hh.Pop_HH`
    :params cutoffs: the list of cutoffs for age categories (e.g.,[5,18,65])
    :type cutoffs: list
    :param cur_year: the current year, for calculating household age.
    :type cur_year: int
    """
    cutoffs = list(cutoffs) + [200]
    counts = np.zeros((100/binsize, len(cutoffs)))

    for k, hh in P.groups['household'].items():
        cur_hh_age = cur_year - P.households[k].founded/365
        cur_hh_age /= binsize
        for i_id in hh:
            cur_age = i_id.age
            i = 0
            while cur_age >= cutoffs[i]:
                i += 1
            counts[cur_hh_age][i] += 1
    return counts


def convert_counts_to_proportions(counts):
    props = np.zeros(counts.shape)
    for i, row in enumerate(counts):
        s = float(sum(row))
        for j, value in enumerate(row):
            props[i][j] = value/s
    return props

def get_hh_comp_dict(P, cutoffs=(5,18,65)):
    """
    Retrieve a snapshot of current household age composisitions (e.g., 
    household A == {2 individuals aged 18-64; 1 individual aged 5-18; 
    2 individuals aged < 5}. NB: this is a snapshot of the population, 
    rather than an aggregate over historical data.

    :params P: the population.
    :type P: :class:`pop_hh.Pop_HH`
    :params cutoffs: the list of cutoffs for age categories (e.g.,[5,18,65])
    :type cutoffs: list
    """
    # hh_cc is a dictionary, keyed by hh size, of unique hh compositions 
    # and the number of times they occur in a population
    cutoffs = list(cutoffs)+[200]
    hh_comps = []
    for hh in P.groups['household'].values():
        cur_comp = np.zeros((len(cutoffs)), dtype=np.int)
        ids = sorted(hh, key=lambda x: x.age, reverse=True)
        for cur_age in [x.age for x in ids]:
            i = 0
            while cur_age >= cutoffs[i]:
                i += 1
            cur_comp[i] += 1
        hh_comps.append(tuple(cur_comp))
    hh_cc = defaultdict(list)
    for hh in set(hh_comps):
        count = hh_comps.count(hh)
        size = sum(hh)
        hh_cc[size].append((hh, count))
    return hh_cc


def compare_hh_type(P, interval=1):
    
    data = np.zeros((4,4))

    for ind in P.I.values():
        if len(ind.hh_type) > (interval+1):
            pre_type = ind.hh_type[-1-interval]
            if pre_type == 'couple_only':
                pre_index = 0
            elif pre_type == 'couple_kids':
                pre_index = 1
            elif pre_type == 'single_kids':
                pre_index = 2
            elif pre_type == 'single_only':
                pre_index = 3
            post_type = ind.hh_type[-1]
            if post_type == 'couple_only':
                post_index = 0
            elif post_type == 'couple_kids':
                post_index = 1
            elif post_type == 'single_kids':
                post_index = 2
            elif post_type == 'single_only':
                post_index = 3

            data[pre_index][post_index] += 1

    rowsums = data.sum(axis=1)
    data /= rowsums.reshape((4,1))
    data *= 100.0
    return data


def process_hh_stats(P):
    
    # get household type stats
    hh_stats_raw = P.hh_type_stats()

    # break down household type stats by age category
    age_bins_a = [15, 25, 35, 45, 55, 65, 75, 101]
    hh_stats = {}
    for k in hh_stats_raw.keys():
        hh_stats[k] = np.histogram([P.I[x].age 
            for x in hh_stats_raw[k]], bins=age_bins_a)[0]

    # normalise by size of category
    age_bins = [(x,y-1) for x,y in zip(age_bins_a[:-1],age_bins_a[1:])]
    age_counts = [len(P.individuals_by_age(x[0], x[1])) for x in age_bins]
    for k, v in hh_stats.items():
        hh_stats[k] = [100.0*x[0]/x[1] for x in zip(v, age_counts)]

    return hh_stats


def sum_hh_stats_ind(P):

    hh_stats_raw = P.hh_type_stats()
    hh_stats = {}
    for k in hh_stats_raw.keys():
        hh_stats[k] = len(hh_stats_raw[k])

    pop_size = len(P.I)

    for k, v in hh_stats.items():
        hh_stats[k] = float(v)/pop_size

    return hh_stats



def process_hh_stats_multi(data):
    """
    Calculate means and stdevs for a list of hh_stat structures.
    Returns a struct of means and a struct of stdevs.
    """

    hh_stat_means = {}
    hh_stat_stdev = {}

    keys = data[0].keys()
    for k in keys:
        hh_stat_means[k] = np.mean(np.array([x[k] for x in data]),0)
        hh_stat_stdev[k] = np.std(np.array([x[k] for x in data]),0)

    return hh_stat_means, hh_stat_stdev


def get_mean_and_stdev(data, last_entry=None):
    """
    get mean and standard deviation of values across a set of distributions
    (e.g., age distribution or hh size distribution)

    data is a list of distribution values
    last_entry is the last real entry; all following entries are summed
        (i.e., for households of size X+)

    returns two lists of means and standard deviations.

    Note: all rows in data must have same number of columns
    """

    data_array = np.array(data)
#    print data_array
    if last_entry:
        for i in range(len(data_array)):
            data_array[i][last_entry-1] = sum(data_array[i][(last_entry-1):])
            data_array[i][(last_entry):] = 0
    data_means = np.mean(data_array, 0)
    data_stdev = np.std(data_array, 0)
    return data_means, data_stdev


def collapse_large_households(data, max_size):
    new_data = []
    for d in data:
        new_data.append(np.concatenate((d[:max_size], [sum(d[max_size:])])))
    return new_data


def get_hh_contact_matrix(P, age_scale=False):
    """
    Basically, for each person, we want to know what aged people they 
    encounter in household contexts.  

    So, start with a matrix of all zeros, from age 0...100 x 0...100.

    Iterate over individuals in population (age = i)

    For each person in their household, other than themselves (age = j), 
    add 1 to matrix(i, j)

    At end, divide all entries by 2 (as we will have counted each 
    "encounter" twice.

    This doesn't precisely give a "rate" as such, because there is no 
    time unit, but it does give a relative level of contact between 
    individuals of different ages.
    """
        
    cm = np.zeros((100,100))

    for ind in P.I.values():
        for cur_hm in P.housemates(ind):
            cm[ind.age][cur_hm.age] += 1

    if age_scale:
        age_dist = np.array(P.age_dist(100,100,norm=False)[0], dtype=np.float32)

        cm /= age_dist
        cm /= age_dist[None].T

    return cm


def create_age_map(cutoffs):

    age_map = []
    prev_cutoff = 0
    for index, cur_cutoff in enumerate(cutoffs):
        age_map.extend([index] * (cur_cutoff - prev_cutoff))
        prev_cutoff = cur_cutoff
    return age_map


def get_hh_contact_matrix_cutoff(P, cutoffs, age_scale=False):
    
    age_map = create_age_map(cutoffs)

    cm = np.zeros((len(cutoffs), len(cutoffs)))

    for ind in P.I.values():
        for cur_hm in P.housemates(ind):
            cm[age_map[ind.age]][age_map[cur_hm.age]] += 1

    if age_scale:
#        print "XXX"

        age_dist = np.array(P.age_dist(
            [0]+cutoffs if cutoffs[0]>0 else cutoffs+[101], 
            cutoffs[-1], norm=False)[0],
                dtype=np.float32)

        ### NB: this [0]+ is problematic if cutoffs already begins with 0

#        print len(age_dist), age_dist
#        print cm.size
#        cm /= age_dist
#        cm /= age_dist[None].T
        cm /= (age_dist * age_dist[None].T)

    return cm

