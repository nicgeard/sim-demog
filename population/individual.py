"""
The base class for an individual.
"""
class Individual(object):
    """
    The base class for an individual.

    :param ID: a unique identifier.
    :type ID: int
    :param age: the individual's age in years.
    :type age: int
    :param sex: the individual's sex (0 == male; 1 == female)
    :type sex: int
    :param adam: `True` if individual is part of the bootstrapped population.
    :type adam: bool
    :param logging: whether or not to write log
    :type logging: bool

    Also stores:

    age_days
      number of days since last birthday.

    groups
      a dictionary keyed by group type.  Values are the ID of the 
      group of that type to which an individual belongs.
     
    partner
      the partner ID of this individual (if any).

    parents
      the parent IDs of this individual (if known).

    children
      the children IDs of this indivudal (if any).

    deps
      the dependent IDs of this individual (if any).

    log
      a list of significant events in this individual's life.

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

    """

    __slots__ = 'ID', 'sex', 'age', 'age_days', 'prev_birth_age', \
            'next_birth_age', 'adam', 'log', 'birth_order', 'groups', \
            'partner', 'divorced', 'parents', \
            'mother', 'father', \
            'children', \
            'deps', 'with_parents'

    def __init__(self, ID, age=0, sex=0, adam=False, logging=True):

        self.ID = ID         
        self.sex = sex
        self.age = age       
        self.age_days = 0    
        self.prev_birth_age = 0
        self.next_birth_age = 0
        self.adam = adam
        if logging: self.log = []
        self.birth_order = 0

        self.groups = {}

        # TODO: refactor to store in 'relations' dictionary
        self.partner = None
        self.divorced = False
        self.parents = []
        self.children = []
        self.deps = []

        # NB: need this if want to store (& compare) hh_type over time
#        self.hh_type = []

        self.with_parents = True


    def set_prev_birth(self, rng):
        """Store the age at which birth has occurred."""
        self.prev_birth_age = self.age * 365 + self.age_days
        ## NB: calculating minimum gap here is more efficient than in can_birth
        ## as only created a single time (per birth)
        self.next_birth_age = self.prev_birth_age + max(270, rng.gauss(365, 90))


    def can_birth(self):
        """True if at least nine months have passed since previous birth."""
        return self.age * 365 + self.age_days - self.next_birth_age > 0


    def add_log(self, time, code, msg, other_id=None):
        """
        Add a log entry.

        :param code: a code for filtering events.
        :type code: string
        :param msg: a text message describing this event.
        :type msg: string
        :param other_id: ID of other individual (if applicable)
        :type other_id: int
        """

        self.log.append({
            'age': self.age, 
            'code': code, 
            'msg': msg, 
            'other': other_id,
            'time': time
            })
