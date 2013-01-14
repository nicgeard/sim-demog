
class Household:

    def __init__(self, time=0, code=None, msg=None, size=None, adam=False):
        self.founded = time
        self.log = []
        self.snapshots = []
        self.adam = adam
        if code != None:
            self.add_log(time, code, msg, size)


    def add_log(self, time, code, msg, size=None):
        if size is None:
            size = self.log[-1]['size']
        self.log.append({
            'age': time - self.founded, 
            'code': code, 
            'msg': msg, 
            'time': time, 
            'size': size 
            })

    
    def size_at_time(self, t):
        """
        On the basis of stored logs, retrieve the household size at the 
        specified time.
        """

        # return -1 if no logs yet, or t occurs before creation of hh
        if not self.log or self.log[0]['time'] > t:
            return -1
        size = self.log[0]['size']
        for l in self.log:
            if l['time'] < t:
                size = l['size']
            else:
                return size
        return self.log[-1]['size']


    def count_events_in_range(self, code, begin, end):
        """
        On the basis of stored logs, retrieve the number of events of type
        'code' that occurred between the specified beginning and end times.
        """
        return len([
            x for x in self.log 
            if x['code'] == code and begin < x['time'] < end])
        

