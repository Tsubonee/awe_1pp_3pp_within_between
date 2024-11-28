# -*- coding: utf-8 -*-
import argparse
import pandas as pd
import numpy as np
import csv
import scipy as sp
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import random
from rpy2 import robjects as R
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, Formula
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import r
pandas2ri.activate()
rstatix = importr('rstatix', lib_loc='./r_packages/')
bootES = importr('bootES', lib_loc='./r_packages/')


gamma_family = r('Gamma(link="log")')

class Statistics:
    def __init__(self, nb_participant, nb_exp, dataframe):
        self.nb_parti = nb_participant
        self.nb_exp = nb_exp
        self.df = dataframe      


    def sampling(self, dataframe, nb):
        df_sampled = dataframe.sample(n = nb)
        return(df_sampled)
    

    def test_GLMM(self, dataframe, target_variable, p_value_index):
        """
        A function to run a mixed-effects model and return the results.

        Parameters:
        - dataframe (pandas.DataFrame): The dataframe to be analyzed.
        - target_variable (str): The target variable for analysis (e.g., 'Awe_S').
        - p_value_index (int): The index of the p-value to use (0 for Intercept, 1 for Perspective, etc.).

        Returns:
        - tuple: The selected p-value and a significance flag.
        """
        # Load lme4 package
        lme4 = importr('lme4')

        # Convert a Python DataFrame to an R DataFrame
        with localconverter(R.default_converter + pandas2ri.converter):
            r_dataframe = R.conversion.py2rpy(dataframe)

        # Define the formula for the GLMM
        formula = Formula(f'{target_variable} ~ Perspective * Scene + (1|ID)')

        print(f"Running GLMM on dataframe with {r_dataframe.nrow} rows and {r_dataframe.ncol} columns")

        try:
            # Fit the GLMM using lme4's glmer function
            model = lme4.glmer(formula, data=r_dataframe, family=gamma_family)
            summary = r.summary(model)

            # Extract fixed effects
            fixed_effects = summary.rx2('coefficients')

            # Convert fixed_effects (R matrix) to numpy array
            with localconverter(R.default_converter + pandas2ri.converter):
                fixed_effects_df = R.conversion.rpy2py(fixed_effects)

            # Extract all p-values
            p_values = fixed_effects_df[:, 3]  # Column 4 corresponds to p-values in R summary

            # Select the specified p-value
            selected_p_value = p_values[p_value_index]

            # Check if the selected p-value is significant
            significant = selected_p_value < 0.05

            return selected_p_value, significant

        except Exception as e:
            print(f"Error in GLMM fitting: {e}")
            return None, False




    def GLMM_ef_size(self, dataframe):
        ef_size = rstatix.wilcox_effsize(data = dataframe, formula = Formula('value ~ Condition'), paired = True)
        return(ef_size)
        

    def testing(self, dataframe):
        z_df, p_df = sp.stats.wilcoxon(dataframe['Synchronous'], dataframe['Asynchronous'])
        if(p_df < 0.05):
            bool = True
        else:
            bool = False
        return(p_df, bool)

    def testing2(self, vector1, vector2):
        z_df, p_df = sp.stats.wilcoxon(vector1, vector2)
        if(p_df < 0.05):
            bool = True
        else:
            bool = False
        return(p_df, bool)

    def testing_2(self, dataframe):
        lm = smf.ols('Embo_Score~Condition',data=dataframe).fit()
        anova_table = sm.stats.anova_lm(lm, typ=2)
        anova_p_value = anova_table.iat[0,3]

    def reshape(self, dataframe):
        df2=dataframe.pivot(index='Participant_ID', columns='Condition', values='Embo_Score')
        return(df2)

    def reshape_long(self, dataframe):
        dataframe["id"] = dataframe.index
        df_long = pd.melt(dataframe, id_vars=['id'], value_vars=['Asynchronous', 'Synchronous'])
        return(df_long)

    def writing(self, file_name, p_value, conformity, ef_size):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.nb_exp, int(self.nb_parti), p_value, conformity, ef_size])

    def wilcox_ef_size(self, dataframe):
        ef_size = rstatix.wilcox_effsize(data = dataframe, formula = Formula('value ~ Condition'), paired = True)
        return(ef_size)