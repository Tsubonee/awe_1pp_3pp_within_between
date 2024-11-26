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
pandas2ri.activate
rstatix = importr('rstatix', lib_loc='./r_packages/')
bootES = importr('bootES', lib_loc='./r_packages/')




class Statistics:
    def __init__(self, nb_participant, nb_exp, dataframe):
        self.nb_parti = nb_participant
        self.nb_exp = nb_exp
        self.df = dataframe      


    def sampling(self, dataframe, nb):
        df_sampled = dataframe.sample(n = nb)
        return(df_sampled)
    

    def test_GLM(self, dataframe, target_variable):
    # """
    # A function to run a mixed-effects model and return the results.

    # Parameters:
    # -dataframe (pandas.DataFrame): The dataframe to be analyzed.
    # -target_variable (str): The target variable for analysis (e.g., 'Awe_S').
    # Returns:

    # tuple: A list of p-values and a significance flag.
    # """
    
    lme4 = importr('lme4')
    
    # Convert a Python DataFrame to an R DataFrame
    with localconverter(R.default_converter + pandas2ri.converter):
        r_dataframe = R.conversion.py2rpy(dataframe)
    
    formula = Formula(f'{target_variable} ~ Perspective * Scene + (1|ID)')
    
    # Create and fit the lme4 model
    model = lme4.glmer(formula, data=r_dataframe, family='Gamma')
    
    # Retrieve the summary results of the model
    summary = R.summary(model)
    
    # Retrieve the summary results of the model
    fixed_effects = summary.rx2('coefficients')
    p_values = fixed_effects.rx(True, 4)  

    significant = any(float(p) < 0.05 for p in p_values)

    return p_values, significant

        

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
