import numpy as np
import pandas as pd
from pandas.core.arrays.datetimelike import dtype_to_unit
from pandas.core.interchange.dataframe_protocol import DataFrame


class BreedingPopulation:
    """
    BreedingPopulation: Takes care of phenotype and relationship matrix
    """

    def __init__(self, phenotype_df: pd.DataFrame, coa_matrix: pd.DataFrame):

        self.phenotype_df = phenotype_df
        self.coa_matrix = coa_matrix
        self.phenotype_df.columns = self.phenotype_df.columns.str.lower()

        self.df = self.phenotype_df.copy()
        self.norm_cols = []

        self.filtered_df = None

    def normalize_traits(self, traits: list, pop_stats: dict):
        for trait in traits:
            trait_name = trait.name
            stats = pop_stats.get(trait_name)

            if stats:
                mean_val = stats['mean']
                std_val = stats['std']

                norm_name = f"norm_{trait_name}"
                # Apply normalization to the main source dataframe
                self.df[norm_name] = (self.df[trait_name] - mean_val) / std_val

                if norm_name not in self.norm_cols:
                    self.norm_cols.append(norm_name)



    def get_values(self, trait_name: str):

        if self.filtered_df is None:
            raise ValueError(f"Pruning has to be done first before you can use this method.")
        return self.filtered_df[trait_name].values


    def do_pruning(self, traits: list):

        """
        This is where I do the pruning by traits.
        :param traits: list of traits
        :return: new df without pruned values
        """

        keep_mask = np.ones(len(self.df), dtype=bool)

        for trait in traits:
            values = self.df[trait.name].values

            prune_mask = (values < trait.lower_prune) | (values > trait.upper_prune)
            keep_mask = keep_mask & ~prune_mask

        self.filtered_df = self.df[keep_mask].copy().reset_index(drop=True)


    def get_trait_stats(self):
        """Returns a dictionary of means and stds for all traits."""
        stats = {}
        for col in self.df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': self.df[col].mean(),
                'std': self.df[col].std() if self.df[col].std() != 0 else 1.0
            }
        return stats
