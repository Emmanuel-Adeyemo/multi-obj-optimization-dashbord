class GAStatus:
    def __init__(self):
        self.active_ga = None      # PyGAD object
        self.optimizer = None      # GeneticAlgorithm class instance
        self.history = []          # list of snapshots
        self.latest_snapshot = {}  # most recent stats here
        self.coa_matrix = None     # uploaded COA is stored here

shared_data_cache = GAStatus()
