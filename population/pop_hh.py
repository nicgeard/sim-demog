"""
A population class in which *households* are a fundamental unit of 
organisation.  
"""
from __future__ import absolute_import
import itertools
import cPickle as pickle
import gzip
from pop_base import Population
from individual import Individual
from household import Household
from utils import sample_table, split_age_probs 

class Pop_HH(Population):
    """
    A population class in which *households* are a fundamental unit of organisation.  

    :param ind_type: the :class:`individual.Individual` (sub)class stored by this population.
    :type ind_type: class
    :param logging: whether or not to write ind/hh logs
    :type logging: bool

    LOG life events for INDIVIDUALS:

    - b: own birth
    - cb: birth of child
    - cd: death of child (dependent)
    - c+: gain of child (relocated)
    - c-: child leaving home
    - d: own death
    - gd: death of parent (guardian)
    - gs: parents (guardians) separating
    - l1: leaving home (single)
    - l2: leaving home (couple)
    - m: marriage (couple formation)
    - s: couple separation
    - r: relocation (as orphan)
    LOG life events for HOUSEHOLDS:

    - f: formation
    - cb: birth of child
    - cd: death of child (dependent)
    - c+: gain of child (relocated)
    - gd: death of adult (guardian)
    - m: merging (due to remarriage)
    - s: separation
    """

    def __init__(self, ind_type=Individual, logging=True):
        super(Pop_HH, self).__init__(ind_type)

        super(Pop_HH, self).init_group_type('household')

        self.logging = logging
        self.households = {}
        self.graveyard = {}


    def duplicate_household(self, t, hh_id):
        """
        Create a duplicate of household hh_id, consisting of 
        new individuals with same household age composition.

        Return id of newly created household.

        :param t: The current time step.
        :type t: int
        :param hh_id: The household to duplicate
        :type hh_id: int
        :returns the id of the newly created household.
        """

        new_hh = []
        for ind in self.groups['household'][hh_id]:
            new_ind = self.add_individual(ind.age, ind.sex, 
                    logging=self.logging)
            if self.logging:
                new_ind.add_log(t, 'i', "Individual (immigrated)")
            new_hh.append(new_ind)
        new_hh_id = self.add_group('household', new_hh)
        self.households[new_hh_id] = Household(t)
        self._init_household(new_hh_id)
        if self.logging:
            self.households[new_hh_id].add_log(t, 'i', "Household (immigrated)",
                    len(self.groups['household'][new_hh_id]))

        return new_hh_id


    def birth(self, t, rng, mother, father=None, sex=0):
        """
        Add a newborn child with specified parents to population.

        By default, the new individual's household is set to that of the first
        parent.

        :param t: The current time step.
        :type t: int
        :param mother: The mother.
        :type mother: Individual
        :param father: The father (if mother is currently partnered).
        :type father: Individual
        :returns: The new individual.

        """

        # create the new individual
        new_ind = self.add_individual(0, sex, logging=self.logging)
        new_ind.birth_order = len(mother.children)+1

        # assign parent and dependency relationships
        mother.set_prev_birth(rng)
        parents = [x for x in (mother, father) if x is not None]
        new_ind.parents = parents
        for x in parents:
            x.children.append(new_ind)
            x.deps.append(new_ind)

        hh_id = parents[0].groups['household']
        self.add_individuals_to_group('household', hh_id, [new_ind])

        if self.logging:
            # record own birth
            new_ind.add_log(t, 'b', "%d born to %s" % (new_ind.ID,
                [x.ID for x in (mother, father) if x is not None]))
            # record children't birth
            for x in parents:
                x.add_log(t, 'c', "Gave birth to %d" % new_ind.ID, new_ind.ID)
            # add the new individual to its parent's household
            self.households[hh_id].add_log(t, 'cb', "Birth of child",
                    len(self.groups['household'][hh_id]))

        return new_ind
    

    def death(self, t, ind):
        """
        Remove individual from population, and return a list of orphaned children.

        :param t: The current time step.
        :type t: int
        :param i_id: The ID of the dead individual.
        :type i_id: int
        """

        # identify individuals who will be orphaned by this death
        orphans = ind.deps if not ind.partner else []

        if self.logging:
            ind.add_log(t, 'd', "Died at age %d" % ind.age)
            if ind.partner:
                ind.partner.add_log(t, 'md', "Partner %d died" % ind.ID, ind.ID)
            for x in ind.deps:
                x.add_log(t, 'gd', "Parent %d died" % ind.ID, ind.ID)
            #self.graveyard[ind.ID] = ind

        # remove as partner
        if ind.partner:
            ind.partner.partner = None

        # remove the dead individual's guardian(s)
        self._remove_from_guardians(t, ind, \
                'cd', "Lost dependent %d (death)" % ind.ID)
 
        # remove dead individual from household and population 
        self._remove_individual_from_hh(t, ind, 'd', "Member died")
        self.remove_individual(ind)

        return orphans


    def process_orphans(self, t, orphans, cutoff, rng):
        """
        Process orphans who result when the last remaining adult guardian
        in their household dies.  If they are above 'adult-age' cutoff, place
        them in a new single household, otherwise, reallocate them to an
        existing family household (with at least one other child).
        """
        for x in orphans:
            if x.age > cutoff:
                self._form_single_hh(t, x)
            else:
                self._reallocate_orphan(t, x, rng)


    def form_couple(self, t, ind, partner):
        """
        Attempt to form a new couple household.  A new couple household is 
        only formed if a suitable partner can be found.

        Return a tuple containing the individual whose household the couple
        now live in (or None if it is a new household), and the household.

        :param t: the current time step.
        :type t: int
        """
    
        assert not ind.partner and not partner.partner
        assert ind.groups['household'] != partner.groups['household']

        # form couple
        ind.partner = partner
        partner.partner = ind

        if self.logging:
            ind.add_log(t, 'm', "Marriage to %d" % partner.ID, partner.ID)
            partner.add_log(t, 'm', "Marriage to %d" % ind.ID, ind.ID)

        # TODO: rather than call sep functions, use this logic to set
        # a target_hh ID variable; if function called with None, then 
        # create a new household, otherwise use this (and don't add
        # existing individuals to moving list.
        if ind.with_parents:
            if partner.with_parents:
                # both individuals live at home, create a new hh
                return None, self._form_couple_hh(t, [ind, partner])
            else: # partner has own hh, move into it
                return partner, self._merge_hh(t, [ind, partner], 
                        partner.groups['household'])
        else: # ind has own hh
            if partner.with_parents or \
                    self.hh_size(ind) > self.hh_size(partner):
                # partner lives at home, or in a smaller household than ind
                return ind, self._merge_hh(t, [partner, ind], 
                        ind.groups['household'])
            else:
                # ind lives in a smaller household than partner
                return partner, self._merge_hh(t, [ind, partner],
                        partner.groups['household'])


    def separate_couple(self, t, ind):
        """
        Separate the couple involving the specified individual, moving their
        partner into a new, single-person household.  Children remain in the 
        original household.

        :param t: The current time step.
        :type t: int
        :param i_id: The individual to separate.
        :type i_id: int
        """

        assert ind.partner

        if ind.sex == 0:
            ind_m = ind; ind_f = ind_m.partner
        else:
            ind_f = ind; ind_m = ind_f.partner

        ind_m.divorced = True; ind_m.partner = None
        ind_f.divorced = True; ind_f.partner = None
        ind_m.deps = []      # ind_f keeps kids!

        # update logs of parents and children
        if self.logging:
            ind_f.add_log(t, 's',
                    "Splitting from %d, staying put" % ind_m.ID, ind_m.ID)
            ind_m.add_log(t, 's',
                    "Splitting from %d, moving out"%ind_f.ID, ind_f.ID)
            for x in ind_f.deps:
                x.add_log(t,'gs',"Parents divorcing, staying with %d"%ind_f.ID)

        self._form_single_hh(t, ind_m)


    def leave_home(self, t, ind):
        """
        Move an individual out of their home, removing them as a dependent of
        their household head(s).

        :param t: the current time step.
        :type t: int
        :param i_id: the ID of the individual to leave home.
        :type i_id: int
        """

        assert not ind.partner
        assert self.hh_size(ind)>1

        self._form_single_hh(t, ind)


    def _remove_from_guardians(self, t, ind, log_code, log_entry, other=None):
        """
        Remove i_id as a dependent upon any of the other individuals in their
        household.  Appends a custom entry to the former guardian's log.

        :param i_id: The ID of the individual leaving their guardians.
        :type i_id: int
        :log_code: The log code classification why the individual is leaving.
        :type log_code: string
        :log_msg: The log message describing why the individual is leaving.
        :type log_msg: string
        :returns: `True` if i_id was dependent upon anyone.
        """

        dep = False
        for x in self.housemates(ind):
            if ind in x.deps:
                x.deps.remove(ind)
                if self.logging:
                   x.add_log(t, log_code, log_entry, other)
                dep = True
        return dep


    def _reallocate_orphan(self, t, ind, rng):
        """
        Move children who have been orphaned by the death of their final
        remaining parent (in the same household) to new, randomly chosen
        households.

        :param t: The current time step.
        :type t: int
        :param i_id: The ID of the orphaned individual.
        :type i_id: int
        :param rng: The random number generator to use.
        :type rng: :class:`random.Random`
        """

        assert not ind.partner 
        assert not ind.children

        cur_hh = ind.groups['household']
        # remove them from their current household
        self._remove_individual_from_hh(t, ind, 'c-', "Lost relocated child")

        # choose destination household from among family households
        candidates = self.groups_by_min_size('household', 3)
        tgt_hh = cur_hh
        while tgt_hh == cur_hh: tgt_hh = rng.sample(candidates, 1)[0]

        # appoint eldest person in new household and partner as guardians
        g_ind = sorted(self.groups['household'][tgt_hh], 
                key=lambda x: x.age)[-1]
        g_ind.deps.append(ind)
        if g_ind.partner:
            g_ind.partner.deps.append(ind)

        # add individual to new household (must happen last!!)
        self.add_individuals_to_group('household', tgt_hh, [ind])

        if self.logging:
            g_ind.add_log(t, 'c+', "Gained dependent %d (orphan)" % ind.ID)
            if g_ind.partner:
                g_ind.partner.add_log(t, 'c+', 
                        "Gained dependent %d (orphan)" % ind.ID)
            self.households[tgt_hh].add_log(t, 'c+', "Gained relocated child",
                    len(self.groups['household'][tgt_hh]))
            ind.add_log(t, 'r', "Relocated - with %d as guardian" % g_ind.ID)


    def _form_single_hh(self, t, ind):
        """
        Move specified individual, with no partner, into their own household.

        :param t: the current time step.
        :type t: int
        :param i_id: the first partner in the couple.
        :type i_id: int
        :returns: the ID of the new household.
        """

        assert not ind.partner
#        assert len(self.housemates(i_id)) > 0

        # remove any guardians
        self._remove_from_guardians(t, ind, \
                'c-', "Lost dependent %d (leaving)" % ind.ID, ind.ID)

        # remove from old household 
        self._remove_individual_from_hh(t, ind, 'l', "Individual left")

        # add to newly created household
        new_hh = self.add_group('household', [ind])

        ind.with_parents = False

        self.households[new_hh] = Household(t)
        if self.logging:
            ind.add_log(t, 'l1', "Leaving household (single)")
            self.households[new_hh].add_log(t, 'f1', "Household formed", 1)

        return new_hh


    def _merge_hh(self, t, inds, hh_id):
        """
        Move p_id and any of their dependents into hh_id.

        :param t: The current time step.
        :type t: int
        :param i_ids: The ids of the new couple
        """

        new_inds = [inds[0]]
        combined_deps = inds[0].deps + inds[1].deps
        inds[0].deps = combined_deps[:]
        inds[1].deps = combined_deps[:]
        new_inds.extend(inds[1].deps)

        inds[0].with_parents = False
        inds[1].with_parents = False

        # remove any guardians of moving partner
        self._remove_from_guardians(t, inds[0], \
                'c-', "Lost dependent %d (leaving)" % inds[0].ID, inds[0].ID)
        
        for ind in new_inds:
            self._remove_individual_from_hh(t, ind, 'l', "Individual left")
       
        self.add_individuals_to_group('household', hh_id, new_inds)

        if self.logging:
            inds[0].add_log(t, 'l2', 
                    "Leaving household - couple with %d"%inds[1].ID, inds[1].ID)
            self.households[hh_id].add_log(t, 'm', \
                    "Household merged (%d individuals)" % len(new_inds),
                    len(self.groups['household'][hh_id]))

        return hh_id


    def _form_couple_hh(self, t, inds):
        """
        Make specified single individuals into a couple and move them into a 
        new household.  Any dependents of either individual accompany them to 
        the new household.

        :param t: the current time step.
        :type t: int
        :param i_ids: the individuals to move into a couple household.
        :type i_ids: list
        :returns: the ID of the new household.
        """

        ind_a = inds[0]; ind_b = inds[1]

        inds[0].with_parents = False
        inds[1].with_parents = False

        # move dependents along with guardians
        new_inds = list(inds)
        combined_deps = ind_a.deps + ind_b.deps
        ind_a.deps = combined_deps[:]
        ind_b.deps = combined_deps[:]
        new_inds.extend(combined_deps)

        # remove any guardians of new couple
        self._remove_from_guardians(t, inds[0], \
                'c-', "Lost dependent %d (leaving)" % inds[0].ID, inds[0].ID)
        self._remove_from_guardians(t, inds[1], \
                'c-', "Lost dependent %d (leaving)" % inds[1].ID, inds[1].ID)
       
        # remove individuals from prior households and create new household
        for ind in new_inds:
            self._remove_individual_from_hh(t, ind, 'l', "Individual left")
        hh_id = self.add_group('household', new_inds)

        self.households[hh_id] = Household(t)

        if self.logging:
            ind_a.add_log(t,'l2',"Leaving household - couple with %d"%ind_b.ID)
            ind_b.add_log(t,'l2',"Leaving household - couple with %d"%ind_a.ID)
            self.households[hh_id].add_log(t, 'f2', \
                    "Household formed (%d individuals)" % \
                    len(new_inds), len(new_inds))

        return hh_id


    def _remove_individual_from_hh(self, t, ind, log_code, log_msg):
        old_hh = self.remove_individual_from_group('household', ind)
        new_size = 0 if old_hh not in self.groups['household'] \
            else len(self.groups['household'][old_hh])

        if self.logging:
            self.households[old_hh].add_log(t, log_code, log_msg, new_size)


    ### Info and helper functions #############################################

    def sample_by_age(self, age, size, rng):
        """DEPRECATED"""
        
        subset = rng.sample(self.I.values(), size) \
                if len(self.I) > size else self.I.values()
        return self.individuals_by_age(age, age, subset)


    def with_parents(self, ind):
        """
        Returns True if i_id lives with his parents (actually, guardians).

        :param i_id: The individual to test.
        :type i_id: int
        :returns: The number of parents with whom ind is living.
        """

        return ind.with_parents

    def num_parents_in_hh(self, ind):
        return_value = 0
        for i in self.hh_members(ind):
            if ind in i.deps:
                return_value += 1
        return return_value


    def print_ind(self, ind):
        """
        Print some information about individual ind_id.
        """

        # TODO: move ind stuff to Ind_HH class.
        print '%d %d %s\n' % \
                (ind.ID, ind.age, ind.groups)

#        print '%d %d %s P(%s %s %s) %d %s D%s H%s\n' % \
#                (ind.ID, ind.age, ind.groups, ind.parents, 
#                self.I.has_key(ind.parents[0]) if ind.parents else -1, 
#                self.I.has_key(ind.parents[1]) if ind.parents else -1, 
#                ind.partner.ID, ind.divorced, ind.deps, 
#                self.hh_members(ind))


    def hh_type_stats(self):
        """
        Aggregates data on the type of household each individual belongs to.

        Current possible household types are:

        - single only
        - couple only
        - single with children
        - couple with children
        - with parents

        Future types should include (at least, maybe):
        
        - group household

        :returns: A dictionary mapping household type to a list of individuals
        belonging to a household of that type.
        """

        stats = {'single_only':[], 'couple_only':[], 
                'single_kids':[], 'couple_kids':[],
                'with_parents':[]}

        for i_id, ind in self.I.iteritems():
            hh = self.groups['household'][ind.groups['household']]
            if len(hh) == 0:
                print "HH of size 0!"
            if len(hh) == 1:
                stats['single_only'].append(i_id)
            else:
                if ind.with_parents:
                    stats['with_parents'].append(i_id)
                else:
                    if ind.partner in set(hh):
                        if len(hh) == 2:
                            stats['couple_only'].append(i_id)
                        else:
                            stats['couple_kids'].append(i_id)
                    else:
                        stats['single_kids'].append(i_id)

        return stats

    
    def get_hh_type(self, ind):
        hh = self.hh_members(ind)
        if len(hh) == 0:
            print "HH of size 0!"
        if len(hh) == 1:
            return 'single_only'
        else:
            num_parents = self.num_parents_in_hh(ind)
            if num_parents == 1:
                return 'single_kids'
            elif num_parents == 2:
                return 'couple_kids'
            else:
                if ind.partner in set(hh):
                    if len(hh) == 2:
                        return 'couple_only'
                    else:
                        return 'couple_kids'
                else:
                    return 'single_kids'



    def sum_hh_stats_group(self):

        hh_stats = {'couple_kids':0, 'couple_only':0, 
                'single_kids':0, 'single_only':0}
        
        for k, cur_hh in self.groups['household'].items():
            hh_type = 'with_parents'
            for cur_ind in cur_hh:
                cur_type = self.get_hh_type(cur_ind)
                if cur_type != 'with_parents':
                    hh_type = cur_type
                
            hh_stats[hh_type] += 1
        del hh_stats['single_only']
        fam_count = sum([v for v in hh_stats.values()])
        for k, v in hh_stats.items():
            hh_stats[k] = float(v)/fam_count

        return hh_stats


    def housemates(self, ind):
        """
        Returns a list of ids of other individuals in i_id's household.

        :param i_id: The individual to test.
        :type i_id: int
        :returns: A list of IDs of other individuals sharing i_id's house.
        """

        return [x for x in \
                self.groups['household'][ind.groups['household']] \
                if x is not ind]


    def hh_members(self, ind):
        """
        Get a list of individuals in i_id's household (including i_id).

        :param i_id: The individual's ID.
        :type i_id: int
        :returns: The size of the individual's household.
        """

        return self.groups['household'][ind.groups['household']]


    def hh_size(self, ind):
        """
        Get the size of an individual's household.

        :param i_id: The individual's ID.
        :type i_id: int
        :returns: The size of the individual's household.
        """

        return len(self.groups['household'][ind.groups['household']])


    def hh_age(self, t, ind):
        """
        Get the age of an individual's household.

        :param t: The current time.
        :type t: int
        :param i_id: The individual's ID.
        :type i_id: int
        :returns: The age of the individual's household.
        """

        return t - self.households[ind.groups['household']].founded


    def hh_parents_siblings(self, ind):
        """
        Get the number of parents/siblings in an individual's household.
        """

        num_parents = self.num_parents_in_hh(ind)
        num_children = 0 if num_parents is 0 else \
                self.hh_size(ind) - num_parents - 1
        return num_parents, num_children
        


    ### Population generation functions #######################################

    def gen_hh_size_structured_pop(self, pop_size, hh_probs, rng):
        """
        Generate a population of individuals with household size structure.

        :param pop_size: The size of the population to be generated.
        :type pop_size: int
        :param hh_probs: A table of household size probabilities.
        :type hh_probs: list
        :param rng: The random number generator to use.
        :type rng: :class:`random.Random`
        """

        i = 0  
        while i < pop_size:
            size = int(sample_table(hh_probs, rng)[0])
            cur_hh = []
            for _ in itertools.repeat(None, size):
                cur_ind = self.add_individual(logging=self.logging)
                cur_hh.append(cur_ind)
                i += 1
            self.add_group('household', cur_hh)


    def gen_hh_age_structured_pop(self, pop_size, hh_probs, age_probs_i,     
            cutoffs, rng):
        """
        Generate a population of individuals with age structure and household 
        composition.

        Household composition here is approximated by the number of 
        individuals who are:

        - pre-school age (0--4)
        - school age (5--17)
        - adult (18+)

        This is a bit ad hoc, but serves current purposes.

        :param pop_size: The size of the population to be generated.
        :type pop_size: int
        :param hh_probs: A table of household size probabilities.
        :type hh_probs: list
        :param age_probs_i: A table of age probabilities.
        :type age_probs_i: list
        :param rng: The random number generator to use.
        :type rng: :class:`random.Random`
        """

        age_probs = [[x, int(y[0])] for x, y in age_probs_i]
        split_probs = split_age_probs(age_probs, cutoffs)
        split_probs.reverse()

        i = 0
        while i < pop_size:
            # get list of [adults, school age, preschool age]
            hh_type = [int(x) for x in sample_table(hh_probs, rng)]
#            hh_type = [int(x) for x in sample_uniform(hh_probs, rng)]
            cur_hh = []
            for cur_hh_type, cur_prob in zip(hh_type, split_probs):
                for _ in itertools.repeat(None, cur_hh_type):
                    sex = rng.randint(0, 1)
                    cur_age = sample_table(cur_prob, rng)
                    cur_ind = self.add_individual(cur_age, sex, 
                            adam=True, logging=self.logging)
                    if self.logging:
                        cur_ind.add_log(0, 'f', "Individual (bootstrap)")
                    cur_hh.append(cur_ind)
                    i += 1
            hh_id = self.add_group('household', cur_hh)
            self.households[hh_id] = Household(0, adam=True)
            if self.logging:
                self.households[hh_id].add_log(0, 'f', "Household (bootstrap)",
                        len(self.groups['household'][hh_id]))


    def gen_single_hh_pop(self, hh_size, rng):
        """
        Generate a dummy population consisting of a single household.
        """
        cur_hh = []
        for i in xrange(hh_size):
            sex = rng.randint(0, 1)
            cur_ind = self.add_individual(20, sex, 
                    adam=True, logging=self.logging)
            if self.logging:
                cur_ind.add_log(0, 'f', "Individual (bootstrap)")
            cur_hh.append(cur_ind)
        hh_id = self.add_group('household', cur_hh)
        self.households[hh_id] = Household(0, adam=True)
        if self.logging:
            self.households[hh_id].add_log(0, 'f', "Household (bootstrap)",
                    len(self.groups['household'][hh_id]))


    def allocate_couples(self):
        """
        For testing/bootstrapping: given a household-structured population,
        for all households containing two or more people who are 18 or older,
        form the two oldest members of the household into a couple and 
        make remaining members dependents of the household head(s).

        There is a rather ugly hack here at the moment to prevent a couple
        being created with a dependent who is the same age.  Initial hh 
        allocation possibly needs to be a touch more elegant.
        """

        for cur_hh in self.groups['household'].keys():
            self._init_household(cur_hh)




    def _init_household(self, cur_hh):
        """
        Bootstrap initial household relationships; making two eldest individuals
        partners (if > 17 years).

        Modifies ages if having two individuals of equal age is likely to cause
        probalems (usually for reallocate orphans).

        Again, a fairly ugly hack at the moment.
        """

        hh_size = len(self.groups['household'][cur_hh])
        if hh_size == 1:
            self.groups['household'][cur_hh][0].with_parents = False
#            print "@ household of size 1"
#            print "#0 (%d)"%(self.groups['household'][cur_hh][0].age), \
#                    self.groups['household'][cur_hh][0].with_parents
        else:
            sorted_by_age = sorted(self.groups['household'][cur_hh],
                    key=lambda x: x.age, reverse=True)
            if hh_size > 2 and sorted_by_age[1].age == sorted_by_age[2].age:
                del self.I_by_age[sorted_by_age[2].age][sorted_by_age[2].ID]
                sorted_by_age[2].age -= 1
                self.I_by_age[sorted_by_age[2].age][sorted_by_age[2].ID] = sorted_by_age[2]
                if self.logging:
                    sorted_by_age[2].log[0]['age'] = sorted_by_age[2].age
            if sorted_by_age[0].age == sorted_by_age[1].age:
                del self.I_by_age[sorted_by_age[0].age][sorted_by_age[0].ID]
                sorted_by_age[0].age += 1
                self.I_by_age[sorted_by_age[0].age][sorted_by_age[0].ID] = sorted_by_age[0]
                if self.logging:
                    sorted_by_age[0].log[0]['age'] = sorted_by_age[0].age
            if sorted_by_age[1].age > 17:
                self._form_couple_no_hh(sorted_by_age[:2])
                sorted_by_age[0].deps = sorted_by_age[2:]
                sorted_by_age[1].deps = sorted_by_age[2:]
                sorted_by_age[0].with_parents = False
                sorted_by_age[1].with_parents = False
            else:
                sorted_by_age[0].deps = sorted_by_age[1:]
                sorted_by_age[0].with_parents = False
#            print "@ household of size", len(sorted_by_age)
#            for i, xx in enumerate(sorted_by_age):
#                print "#%d -- %d -- (%d)"%(i, xx.ID, xx.age), xx.with_parents, 
#                if xx.partner:
#                    print xx.partner.ID
#                else:
#                    print ""



    def _form_couple_no_hh(self, inds):
        """
        For testing/bootstrapping: forms a couple (as above), but sets 
        ind.partner fields without modifying households.
        """

        assert inds[0].groups['household'] == inds[1].groups['household']

        inds[0].partner = inds[1]
        inds[1].partner = inds[0]


    def save(self, filepath):
        self.flatten()
        f = gzip.GzipFile(filepath, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        self.lift()


    @classmethod
    def load(self, filepath):
        f = gzip.GzipFile(filepath, 'rb')
        pop = pickle.load(f)
        pop.lift()
        return pop


    def flatten(self):
        """
        For pickling... flatten recursive references.
        """
        for ind in self.I.values() + self.graveyard.values():
            if ind.partner:
                ind.partner = ind.partner.ID
            if ind.parents:
                ind.parents = [x.ID for x in ind.parents]
            if ind.children:
                ind.children = [x.ID for x in ind.children]
            if ind.deps:
                ind.deps = [x.ID for x in ind.deps]


    def lift(self):
        """
        For unpickling... replace recursive references.
        """
        for ind in self.I.values() + self.graveyard.values():
            if ind.partner:
                if ind.partner in self.I:
                    ind.partner = self.I[ind.partner]
                else:
                    ind.partner = self.graveyard[ind.partner]
            ind.parents = self.lift_list(ind.parents)
            ind.children = self.lift_list(ind.children)
            ind.deps = self.lift_list(ind.deps)


    def lift_list(self, data):
        lifted_data = []
        for i in data:
            if i in self.I:
                lifted_data.append(self.I[i])
            elif i in self.graveyard:
                lifted_data.append(self.graveyard[i])
        return lifted_data


#    def immigration(self, t, age=20, sex=0):
#        """
#        Add a new (infected) immigrant - 
#        NB: not currently used
#        """
#
#        new_id = self.add_individual(age, sex)
#        if self.logging:
#            self.I[new_id].add_log(t,'i',"%d immigrated to population"%new_id)
#        new_hh = self.add_group('household', [new_id])
#        self.households[new_hh] = Household(t, 'im', \
#                "Household formed (migrant %s, aged %d" \
#                % ('m' if sex==0 else 'f', age), 1)
#        return new_id
    


