import numpy as np
import pandas as pd
import pygad
import time
from TraitRules import TraitRules

class GeneticAlgorithm:

    def __init__(self, breeding_pop, primary_trait, sec_trait_list, selection_size=100, max_parent_use=5,
                  div_target=0.2, div_penalty=1.2, primary_weight=2.0):

        pop_stats = breeding_pop.get_trait_stats()


        for trait in sec_trait_list:
            trait.finalize_thresholds(pop_stats)

        breeding_pop.normalize_traits(sec_trait_list, pop_stats)
        # print(f"original df columns after norm: {breeding_pop.df.columns.tolist()}")

        # if breeding_pop.filtered_df is None:
        breeding_pop.do_pruning(sec_trait_list)

        # print(f"filtered df columns after pruning: {breeding_pop.filtered_df.columns.tolist()}")


        self.breeding_pop = breeding_pop
        self.primary_trait = primary_trait
        self.sec_trait_list = sec_trait_list
        self.selection_size = selection_size
        self.max_parent_use = max_parent_use - 1
        # self.parent_overuse_penalty = parent_overuse_penalty
        self.div_target = div_target
        self.div_penalty = div_penalty
        # self.div_bonus = div_bonus
        self.primary_weight = primary_weight
        self.stop_signal = False
        self.is_paused = False



    def fitness_function(self, ga_instance, solution, solution_idx):

        # x'y1/x'x - sum(pk/x'x) - p'Cp * e * d

        selected_dta = self.breeding_pop.filtered_df.iloc[solution]
        # print(self.breeding_pop.filtered_df.columns)

        mean_primary = np.mean(selected_dta[f'norm_{self.primary_trait}'])
        weighted_primary = mean_primary * self.primary_weight


        n = len(selected_dta)
        # primary trait

        # secondary trait penalties
        total_sec_penalties = 0
        for trait in self.sec_trait_list:
            vals = selected_dta[f'norm_{trait.name}'].values
            # trait_std = np.std(self.breeding_pop.filtered_df[trait.name])
            total_sec_penalties += trait.penalty_logic(vals)

        # parental usage - exponential penalty
        selected_parents = pd.concat([selected_dta['parent1'], selected_dta['parent2']])
        counts = selected_parents.value_counts()

        valid_pars = [p for p in counts.index if p in self.breeding_pop.coa_matrix.index]

        total_slots = len(selected_parents)
        p_vec = np.array([counts[p] / total_slots for p in valid_pars])

        rel_mat = self.breeding_pop.coa_matrix.loc[valid_pars, valid_pars].values

        group_kinship = p_vec.T @ rel_mat @ p_vec

        mask = np.eye(rel_mat.shape[0], dtype=bool)
        max_pairwise = np.max(rel_mat[~mask]) if rel_mat.size > 1 else 0
        e = 1 if max_pairwise >= (1 - self.div_target) else 0

        counts = selected_parents.value_counts()
        max_count = counts.max()
        excess_use = ((max_count - self.max_parent_use) / self.max_parent_use)**2

        # excess_use = 1 if counts.max() > self.max_parent_use else 0
        div_trigger = e + excess_use

        # div_trigger = 1 if (e == 1 or excess_use == 1) else 0

        diversity_penalty = group_kinship * div_trigger * self.div_penalty
        # diversity_penalty = group_kinship * e * excess_use * self.div_penalty

        # final objective function
        fitness = weighted_primary - total_sec_penalties - diversity_penalty
        # print(e, excess_use, diversity_penalty, fitness)

        return fitness

    # gemini suggested this to allow data capture across pages
    def create_snapshot(self, ga_instance):
        """
        Standardized way to pull stats for the Dash UI
        """
        # gets the current best solution from this generation
        sol, fitness, _ = ga_instance.best_solution()
        selected_dta = self.breeding_pop.filtered_df.iloc[sol]

        # calc div
        parents = pd.concat([selected_dta['parent1'], selected_dta['parent2']]).unique()
        valid = [p for p in parents if p in self.breeding_pop.coa_matrix.index]

        div = 0
        if len(valid) > 1:
            small_coa = self.breeding_pop.coa_matrix.loc[valid, valid]
            # take out diagonals from sum
            off_diag_sum = small_coa.values.sum() - len(valid)
            # divide by number of off-diagonal elements
            avg_coa = off_diag_sum / (len(valid) ** 2 - len(valid))
            div = 1 - avg_coa

        # safety check for fitness make thread no crash
        last_fit = ga_instance.last_generation_fitness
        mean_fit = np.mean(last_fit) if last_fit is not None else fitness

        # standardized selection differential - deltaG. I used original pre-pruned df
        pop_mean = self.breeding_pop.df[self.primary_trait].mean()
        pop_std = self.breeding_pop.df[self.primary_trait].std()
        current_mean = selected_dta[self.primary_trait].mean()
        dg = (current_mean - pop_mean) / pop_std if pop_std != 0 else 0

        return {
            'iteration': ga_instance.generations_completed,
            'max_gen': ga_instance.num_generations,
            'best_fitness': fitness,
            'mean_fitness': mean_fit,
            'diversity': div,
            'mean_yield': current_mean,
            'delta_g': dg,
            'selected_ids': selected_dta.index.tolist()
        }
