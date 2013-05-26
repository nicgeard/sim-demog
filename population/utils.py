import os,sys
import numpy as np
from configobj import ConfigObj
from validate import Validator
import Image
from math import exp,log
from collections import defaultdict

def geomean(nums):
    ml = sum([log(x) for x in nums])
    return exp(ml/len(nums))
#    return (reduce(lambda x, y: x*y, nums))**(1.0/len(nums))

def adjust_prob(prob, t_dur):
    N = t_dur/365.0
    return 1.0 - pow(1.0-prob, N)


def create_thumbnail(filename, size):
    im = Image.open(filename)
    pathname = os.path.split(filename)
    outpath = os.path.join(pathname[0], 'thumbnails')
    create_path(outpath) 
    outfile = os.path.splitext(pathname[1])[0] + '.thumbnail'
    im.thumbnail(size, Image.ANTIALIAS)
    im.save(os.path.join(outpath, outfile), "JPEG")
    return os.path.join('thumbnails', outfile)


def parse_params(filename='params.cfg', specfile='paramspec.cfg'):
    """
    Parse and validate parameter file, and return dictionary of param values.
    """

    params = ConfigObj(filename, configspec=specfile)
    val = Validator()
    params.validate(val)
    return params


def merge_params(*src_params):
    params = ConfigObj()
    for p in src_params:
        params.merge(p)
        params.inline_comments = dict(params.inline_comments, 
                **p.inline_comments)
    return params


def create_path(path_name):
    """
    Create a path if it doesn't already exist.
    """

    if not os.path.exists(path_name):
        os.makedirs(path_name)


def sample(probs, rng):
    """
    Returns i E [0, len(probs)-1] with probability probs[i]
    """
    
    x = rng.random();  prob_sum = 0.0
    for i in xrange(0, len(probs)):
        prob_sum += probs[i]
        if prob_sum >= x:
            return i
    # In almost all cases, this loop should return before completing,
    # as sum(probs) == 1.0 (which is >= x).  If x is very close to 1.0 
    # however, rounding errors in summing probabilities may mean that     
    # x>sum(probs). Therefore, return final index if this occurs:
    return len(probs) - 1       


def sample_table(table, rng):
    """
    Given a table of [p, x], sample and return event x with probability p
    """

    i = sample(zip(*table)[0], rng)
#    print i
    return table[i][1]


def sample_uniform(table, rng):
    i = rng.randint(0, len(table)-1)
    return table[i][1]


def load_probs_new(fname, sorted=False):
    """
    Return an arbitrary probability table loaded from file fname

    The table has the form:
    prob_0, [list of data_values_0]
    prob_1, [list of data_values_1]
    ...

    A check is made to ensure that probabilities sum to 1.0

    probabilities are sorted from most to least frequent for future efficiency

    lines beginning with '#' are skipped (comments)

    lines beginning with '@' force the beginning of a new subtable
    """

    t = []
    current_t = []
    for l in open(fname, 'r'):
        if l[0] is '#': continue
        if l[0] is '@':
            t.append(current_t[:])
            current_t = []
        else:
            line = l.strip().split(' ')
            current_t.append([float(line[0]), line[1:]])
    t.append(current_t[:])

    for current_t in t:
        if abs(sum(zip(*current_t)[0])-1.0) > 0.00001:
            sys.stderr.write("Probs in file %s don't sum to 1.0" % fname)
            exit()

        if sorted: current_t.sort(reverse=True)

    if len(t) == 1:
        return t[0]
    else:
        return t


def load_prob_tables(fname):
    """
    Load a set of probability tables of the form:
    data_value, prob_0, prob_1, prob_2, ...
    ...

    Used for time-varying probabilities (i.e., fertility by age).

    Returns a dictionary keyed by data_value, with a normal 
    prob_0, [data_value] table (as below).

    This isn't particularly optimal, but allows current sampling functions 
    to be used unchanged.
    """

    d = defaultdict(list)
    for l in open (fname, 'r'):
        if l[0] is '#': continue
        line = l.strip().split(' ')
        for i, val in enumerate(line[1:]):
            d[int(i)].append([float(val), [int(line[0])]])

    return d


def load_probs(fname, sorted=False):
    """
    Return an arbitrary probability table loaded from file fname

    The table has the form:
    prob_0, [list of data_values_0]
    prob_1, [list of data_values_1]
    ...

    A check is made to ensure that probabilities sum to 1.0

    probabilities are sorted from most to least frequent for future efficiency
    """

    t = []
    for l in open(fname, 'r'):
        if l[0] is '#': continue
        line = l.strip().split(' ')
        t.append([float(line[0]), line[1:]])

    if abs(sum(zip(*t)[0])-1.0) > 0.00001:
        stderr.write("Probs in file %s don't sum to 1.0") % fname; exit(1)

    if sorted: t.sort(reverse=True)

    return t



def load_age_rates(fname):
    """
    Load age-dependent rates from file fname.

    File has the format:
    age rate_1 rate_2 rate_3 ... etc
    """

    t = []
    for l in open(fname, 'r'):
        if l[0] is '#': continue
        t.append([eval(x) for x in l.strip().split(' ')])
    return t



def load_age_rates_d(fname):
    """
    Load age-dependent rates from file fname.  Returns as dictionary

    File has the format:
    age rate_1
    """

    t = {}
    for l in open(fname, 'r'):
        if l[0] is '#': continue
        t[eval(l[0])] = [eval(x) for x in l.strip().split(' ')[1:]]
    return t



def load_age_rates_range(fname):
    """
    Load age-dependent rates from file fname

    File has the format:
    [bound_a, bound_b] rate_1 rate_2 rate_3 ... rate_n
    """

    t = []
    for l in open(fname, 'r'):
        if l[0] is '#': continue
        line = l.strip().split(' ')
        r = eval(line[0])
        del line[0]
        for x in xrange(r[0], r[1]+1):
            t.append([x]+[float(y) for y in line])
    return t


def load_prob_list(fname):
    """
    Loads a sequence of probabilities/rates and returns them as a list.

    For use in specifying, e.g., time-varying marriage rates.

    File has the format:
    value_1
    value_2
    value_3
    ...
    """
    t = []
    for l in open(fname, 'r'):
        if l[0] is '#': continue
        line = l.strip().split(' ')
        r = eval(line[0])
        t.append(float(line[0]))
    return t


def split_age_probs(age_probs, cutoffs):
    """
    Split age distributions into rescaled distributions over subranges 
    specified by cutoffs.

    e.g., if initial age distribution is over range [0, 100] and cutoffs are
    [5, 18], then the function will return three distributions, with ranges
    [0, 4], [5, 17] and [18, 100], with probabilities summing to 1.0 for 
    each distribution.
    """

    cutoffs.append(int(age_probs[-1][1])+1)
    ranges = [];  bound_a = 0
    for bound_b in cutoffs:
        cur_range = []
        probsum = 0.0
        for j in range(bound_a, bound_b):
            probsum += age_probs[j][0]
        for j in range(bound_a, bound_b):
            cur_range.append([age_probs[j][0]/probsum, j])
        ranges.append(cur_range)
        bound_a = bound_b
    return ranges



def parse_hh_data(ifname, ofname):
    """
    Parse the ABS data set to create a 3-dimensional array in which entry
    i,j,k contains the probability of a household with i adults, j school-age
    children and k preschool-age children.

    These probabilities are then written to a file in the format:
    prob i j k
    ...
    for each non-zero probability
    """

    f = open(ifname, 'r')

    rawdata = []
    for l in f.readlines()[2:]:
        # all instances of '10+' become '10'
        l2 = [int(x.strip('+')) for x in l.split()]
        # remove syd and melb columns, collapse working and non-working
        l3 = l2[:2]+[l2[2]+l2[3]]+[l2[6]]
        # collapse all occurrences of >10 to 10
        if l3[0] >= 10: l3[0] = 10; 
        if l3[1] >= 10: l3[1] = 10; 
        if l3[2] >= 10: l3[2] = 10; 
        rawdata.append(l3)
    f.close()

    # collapse into a 3 dimensional matrix (preschool, school, adult)
    data = np.zeros([11,11,11])
    list_sum = 0
    for l4 in rawdata:
        list_sum += l4[3]
        data[l4[2]][l4[1]][l4[0]] += l4[3]

    # convert to probabilities
    data /= data.sum()

    f = open(ofname, 'w')
    for i in range(11):
        for j in range(11):
            for k in range(11):
                if data[i][j][k] > 0:
                    f.write("%g %d %d %d\n" % (data[i][j][k], i, j, k))
    f.close()

