import numpy as np


class TraitRules:

    def __init__(self, name, direction='high', lower_prune=None,
                 upper_prune=None, lower_thresh=None, upper_thresh=None, penalty_weight=0.0):

        self.name = name
        self.direction=direction # normally, yield is the primary trait, but this allows flexibility for other traits to be focus
        # TODO: implement direction in app. Present direction is to maximize fitness.
        #  If lower is better for primary trait, direction='low' and fitness will be minimized.


        self.lower_prune = lower_prune if lower_prune is not None else -np.inf
        self.upper_prune = upper_prune if upper_prune is not None else np.inf
        self.lower_thresh = lower_thresh
        self.upper_thresh = upper_thresh
        self.penalty_weight = penalty_weight

        self.norm_lower = None
        self.norm_upper = None

    def finalize_thresholds(self, pop_stats):
        """Z-scores for raw scores"""
        stats = pop_stats.get(self.name)
        if stats:
            if self.lower_thresh is not None:
                self.norm_lower = (self.lower_thresh - stats['mean']) / stats['std']
            if self.upper_thresh is not None:
                self.norm_upper = (self.upper_thresh - stats['mean']) / stats['std']


    def penalty_logic(self, values):

        # s/sigma
        total_penalty = 0.0

        vals = np.array(values, dtype=float)

        u_thresh = float(self.norm_upper) if self.norm_upper is not None else None
        l_thresh = float(self.norm_lower) if self.norm_lower is not None else None

        # for traits where higher is better eg yield
        if l_thresh is not None:
            low_mask = vals < l_thresh
            if np.any(low_mask):
                diffs = l_thresh - vals[low_mask]

                # print(np.sum(diffs * self.penalty_weight))
                total_penalty += np.sum(diffs * self.penalty_weight)

        # for traits where lower is better eg plant height
        if u_thresh is not None:
            high_mask = vals > u_thresh
            if np.any(high_mask):

                diffs = vals[high_mask] - u_thresh

                # debug check to make sure I no mess up the penalty calc for the diff thresholds
                if np.any(diffs < 0):
                    print(f"Found negative diff in {self.name}!")
                    print(f"Threshold used: {u_thresh}")
                    print(f"Bad Values: {vals[high_mask][diffs < 0]}")

                # print(np.sum(diffs * self.penalty_weight))
                total_penalty += np.sum(diffs * self.penalty_weight)

        return total_penalty/len(values)
