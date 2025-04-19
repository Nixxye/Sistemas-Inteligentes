class TSolution:
    def __init__(self, temperature = None, alpha = None, distance = None):
        self.Temperature = temperature
        self.Alpha = alpha
        self.Distance = distance
        self.Path = []
        pass

class GASolution:
    def __init__(self, path = [], slice = None, distance = None):
        self.Slice = slice
        self.Distance = distance
        self.Path = path
        pass
