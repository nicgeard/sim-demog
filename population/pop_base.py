"""
The base class for a population.
"""
from __future__ import absolute_import
import numpy as np
import itertools
from population.individual import Individual
from utils.utils import sample_table
from collections import defaultdict

class Population(object):
    """
    The base class for a population containing objects of type 
    :class:`individual.Individual`, or subclass.  
    
    :class:`population.Population` is the simplest population class, providing
    sufficent functionality for age-structured populations and arbitrary
    contact groups.  While it provides the basic functionality necessary for
    implementing households and demographic processes, it stops short of
    implementing these explicitly.

    A derived class, :class:`pop_hh.Pop_HH`,  implements households (a type of
    contact group) and associated functionality.

    :param ind_type: the :class:`individual.Individual` (sub)class stored by this population.
    :type ind_type: class

    .. note:: 

        An issue arises when an individual is added to two groups of the same
        type; their group membership is changed, but the membership list of
        their old group is NOT updated.  Need to decide how best to enforce
        this.
    
    """

    def __init__(self, ind_type=Individual): 
        self.ind_type = ind_type

        # A dictionary of individuals keyed by their unique ID.
        self.I = {}

        # A dictionary of individual IDs keyed by age
        # This uses some space, but provides much more rapid access when trying
        # to select individuals by age (e.g., as mothers or partners)
        self.I_by_age = defaultdict(dict)

        # A nested dictionary containing contact group memberships.  The first
        # key specifies the type of group (a string) and the second key 
        # specifies the group's ID.  The value is a list of IDs of individuals 
        # belonging to that group.
        self.groups = {}
        
        # A dictionary of counters storing the next available ID for 
        # individuals and group types.
        self.next_id = {'individual': 0}


### basic individual and group operations ###

    def add_individual(self, age=-1, sex=0, adam=False, logging=True):
        """
        Add a new individual to the population.

        :param age: The age of the newly added individual.
        :type age: int
        :param adam: ``True`` if individual is created as part of bootstrap population.
        :type adam: bool
        :returns: indtype -- the newly added individual.
        """

        new_id = self.next_id['individual']
        self.I[new_id] = self.ind_type(new_id, age, sex, adam, logging)
        self.I_by_age[age][new_id] = self.I[new_id]
        self.next_id['individual'] += 1
        return self.I[new_id]


    def remove_individual(self, ind):
        """
        Remove an individual from the population and, by extension, from all
        of their groups.

        :param ind: The individual to be removed.
        :type i_id: indtype
        """

        for x in ind.groups.keys():
            self.remove_individual_from_group(x, ind)
        del self.I_by_age[ind.age][ind.ID]
        self.I.pop(ind.ID)


    def init_group_type(self, group_type):
        """
        Add a new type of group to the population.

        :param group_type: The label used to denote this group type.
        :type group_type: string
        """

        assert not self.groups.has_key(group_type)

        self.groups[group_type] = {}
        self.next_id[group_type] = 0


    def add_group(self, group_type, inds=None):
        """
        Add a new contact group of specified type to the population and
        return its ID.  Optionally, specify a set of indivudals to be added to 
        the newly created group.

        :param group_type: The type of group to add.
        :type group_type: string
        :param i_ids: IDs of individuals to add to the group.
        :type i_ids: list
        :returns: The ID of the newly created group.

        .. note::
            The group type must already have been initialised using 
            :func:`init_group_type`.

        """

        assert self.groups.has_key(group_type)

        new_id = self.next_id[group_type]
        self.groups[group_type][new_id] = []
        if inds:
            self.add_individuals_to_group(group_type, new_id, inds)
        self.next_id[group_type] += 1
        return new_id


    def remove_group(self, group_type, group_id):
        """
        Remove an existing contact group.

        :param group_type: The type of the group to be removed.
        :type group_type: string
        :param group_id: The ID of the group to be removed.
        :type group_id: int
        """

        assert self.groups[group_type].has_key(group_id)

        for x in self.groups[group_type][group_id]:
            self.I[x.ID].groups.pop(group_type)
        self.groups[group_type].pop(group_id)


    def add_individuals_to_group(self, group_type, group_id, inds):
        """
        Add individuals to an existing group.

        :param group_type: The type of the group.
        :type group_type: string
        :param group_id: The ID of the group.
        :type group_id: int
        :param i_ids: IDs of individuals to add to the group.
        :type i_ids: list

        .. note::
            
            No checking is done to ensure that individuals are not already a
            member of a group of this type.

        """

        assert self.groups[group_type].has_key(group_id)

        self.groups[group_type][group_id].extend(inds)
        for x in inds:
            x.groups.__setitem__(group_type, group_id)


    def remove_individual_from_group(self, group_type, ind):
        """
        Remove an individual from a group to which they belong.  Additionally,
        remove the group if it now contains no members.

        :param group_type: The type of the group the individual is to leave.
        :type group_type: int
        :param i_id: The ID of the individual to be removed.
        :type i_id: int
        :returns: The ID of the group the individual has left.
        """

        assert ind.groups.has_key(group_type)

        group_id = ind.groups[group_type]
        ind.groups.pop(group_type)
        self.groups[group_type][group_id].remove(ind)
        # remove group if now empty
        if len(self.groups[group_type][group_id]) <= 0:
            self.groups[group_type].pop(group_id)  
        return group_id


    
### update age of population ##########

    def age_population(self, period):
        """
        Age each individual in population by duration specified (in days).

        :param days: The number of days to age the population.
        :type days: int
        """

        birthdays = defaultdict(list)

        for ind in self.I.itervalues():
            ind.age_days += period
            if ind.age_days >= 364:   # i has a birthday
                del self.I_by_age[ind.age][ind.ID]
                ind.age += 1
                ind.age_days %= 364
                birthdays[ind.age-1].append(ind.ID)
                self.I_by_age[ind.age][ind.ID] = self.I[ind.ID]
    
        return birthdays


### access individuals and groups by age, size, etc. ##########

    def ind_ids_by_age(self, min_age, max_age, subset=None):
        if min_age == max_age:
            return self.I_by_age[min_age].keys()
        else:
            x = []
            for cur_age in xrange(min_age, max_age+1):
                x += self.I_by_age[cur_age]
            return x


    def individuals_by_age(self, min_age, max_age, subset=None):
        """
        Return a list of IDs of individuals in the specified age range.

        :param min_age: The minimum age to include.
        :type min_age: int
        :param max_age: The maximum age to include.
        :type max_age: int
        :returns: a list of IDs of individuals in age range.
        """

        if min_age == max_age:
            return self.I_by_age[min_age].values()
        else:
            x = []
            for cur_age in xrange(min_age, max_age+1):
                x += self.I_by_age[cur_age]
            return x

#       OLD version -- much, much slower!
#        if not subset:
#            return [x.ID for x in self.I.values() \
#                    if min_age <= x.age <= max_age]
#        else:
#            return [x.ID for x in subset \
#                    if min_age <= x.age <= max_age]


    def individuals_by_group_size(self, group_type, size):
        """
        Return a list of IDs of individuals in groups of specified type and
        size.

        :param group_type: The type of groups to aggregate.
        :type group_type: string
        :param size: The size of groups to aggregate.
        :type size: int
        :returns: A list of IDs of individuals in appropriately sized groups.

        """

        return [x for y in [grp for grp in self.groups[group_type].values() \
                if len(grp) == size] for x in y]


    def individuals_by_min_group_size(self, group_type, size):
        """
        Return a list of IDs of individuals in groups of specified type and
        minimum size.

        :param group_type: The type of groups to aggregate.
        :type group_type: string
        :param size: The size of groups to aggregate.
        :type size: int
        :returns: A list of IDs of individuals in appropriately sized groups.

        """

        return [x for y in [grp for grp in self.groups[group_type].values() \
                if len(grp) >= size] for x in y]


    def groups_by_size(self, group_type, size):
        """
        Return a list of IDs of groups that are of the specified type and size.

        :param group_type: The type of groups to evaluate.
        :type group_type: string
        :param size: The size of groups to evaluate.
        :type size: int
        :returns: A list of IDs of groups of appropriate size.
        """

        return [x for x in self.groups[group_type].keys() \
                if len(self.groups[group_type][x]) == size]


    def groups_by_min_size(self, group_type, size):
        """
        Return a list of IDs of groups that are of the specified type and 
        *at least* the specified size.

        :param group_type: The type of groups to evaluate.
        :type group_type: string
        :param size: The minimum size of groups to evaluate.
        :type size: int
        :returns: A list of IDs of groups of at least size.
        """

        return [x for x in self.groups[group_type].keys() \
                if len(self. groups[group_type][x]) >= size]



### initialisation functions for setting up age and group structures ##########

    def gen_age_structured_pop(self, pop_size, age_probs, rng):
        """
        Generate a population of individuals with given age structure.

        :param pop_size: The number of individuals to generate.
        :type pop_size: int
        :param age_probs: A table mapping probabilities to age.
        :type age_probs: list
        :param rng: The random number generator to use.
        :type rng: :class:`random.Random`
        """

        # TODO: move out of class?

        for _ in itertools.repeat(None, pop_size):
            self.add_individual(age=int(sample_table(age_probs, rng)[0]))


    def allocate_groups_by_age(self, group_type, 
            size_probs, min_age, max_age, rng):
        """
        Allocate individuals in a given age range to groups
        with a given size distribution.

        :param group_type: The type of groups to create.
        :type group_type: string
        :param size_probs: A table mapping probability to group size.
        :type size_probs: list
        :param min_age: The minimum age to include.
        :type min_age: int
        :param max_age: The maximum age to include.
        :type max_age: int
        :param rng: The random number generator to use.
        :type rng: :class:`random.Random`

        """

        # TODO: move out of class? FIX ind/ID

        assert not self.groups.has_key(group_type)

        self.init_group_type(group_type)

        inds = self.individuals_by_age(min_age, max_age)
        rng.shuffle(inds)
        while len(inds) > 0:
            size = int(sample_table(size_probs, rng)[0])
            members = [x for x in inds[:size]]
            group_id = self.add_group(group_type, members)
            for x in members:
                x.groups.__setitem__(group_type, group_id)
            inds = inds[size:]

#        ids = self.individuals_by_age(min_age, max_age)
#        rng.shuffle(ids)
#        while len(ids) > 0:
#            size = int(sample_table(size_probs, rng)[0])
#            members = [self.I[x] for x in ids[:size]]
#            group_id = self.add_group(group_type, members)
#            [x.groups.__setitem__(group_type, group_id) for x in members]
#            del ids[:size]


    def allocate_groups_by_group(self, target_type, source_type, 
            size_probs, rng):
        """
        Create 'groups of groups' by randomly aggregating lower level groups
        into higher level groups according to the given size distribution.

        For example, build neighbourhoods out of households by
        grouping households according to some distribution over number of 
        households per neighbourhood.

        :param target_type: The type of groups to create.
        :type target_type: int
        :param source_type: The type of groups to aggregate.
        :type source_type: int
        :param size_probs: A table mapping probability to group size.
        :type size_probs: list
        :param rng: The random number generator to use.
        :type rng: :class:`random.Random`

        """

        # TODO: factor out

        assert self.groups.has_key(source_type)
        assert not self.groups.has_key(target_type)
        
        self.init_group_type(target_type)

        ids = self.groups[source_type].keys()
        rng.shuffle(ids)
        while len(ids) > 0:
            size = int(sample_table(size_probs, rng)[0])
            members = []
            group_id = self.add_group(target_type, members)
            for source_id in ids[:size]:
                for i in self.groups[source_type][source_id]:
                    self.I[i].groups[target_type] = group_id
                    members.append(self.I[i])
                self.add_individuals_to_group(target_type, group_id, members)
            del ids[:size]



### measure statistical age and group size distributions ##########

    def age_dist(self, num_bins=101, max_age=101, norm=True):
        """
        Return the age distribution of a population.

        :param num_bins: the number of bins to group the population into.
        :type num_bins: int
        :param max_age: the maximum possible age.
        :type max_age: int
        :returns: a tuple containing a list of values and a list of bin edges.
        """

        # TODO: factor out (replace with get_age_list())

        ages = [i.age for i in self.I.values()]
        return np.histogram(ages, bins=num_bins, \
                range=(0, max_age), normed=norm)


    def group_size_dist(self, group_type, max_size=10, norm=True):
        """
        Return the size distribution of groups of specified type.

        :param group_type: the type of group to evaluate.
        :type group_type: string
        :param max_size: the maximum possible group size.
        :type max_size: int
        :returns: a tuple containing a list of values and a list of bin edges.
        """

        # TODO: factor out (replace with get_group_size_list())

        sizes = [len(hh) for hh in self.groups[group_type].values()]
        return np.histogram(sizes, bins=max_size, \
                range=(1,max_size+1), normed=norm)


    def group_size_avg(self, group_type):
        """
        Return the average size of groups of specified type.
        """

        sizes = [len(hh) for hh in self.groups[group_type].values()]
        return np.mean(sizes)

    
    def individuals_by_group_size_dist(self, group_type, 
            max_size=10, norm=True):
        size_dist, bins = self.group_size_dist(group_type, max_size, norm)
        dist = [i*x for i,x in enumerate(size_dist)]
        return dist, bins


    def print_population_summary(self):
        """
        A simple info dump.
        """

        print len(self.I), 'individuals'
        for k in self.groups.keys():
            cur_len = len(self.groups[k])
            cur_avg = sum([len(i) for i in self.groups[k].values()]) /\
                    float(cur_len)
            print '%d %s%s, mean size = %g' % \
                    (cur_len, k, 's' if cur_len>1 else '', cur_avg)


