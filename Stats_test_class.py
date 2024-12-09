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

        try:
            # Convert Perspective and Scene to factors in R
            r_dataframe = r('''
            function(df) {
                df$Perspective <- as.factor(df$Perspective)
                df$Scene <- as.factor(df$Scene)
                return(df)
            }
            ''')(r_dataframe)
        except Exception as e:
            print(f"Error in converting variables to factors: {e}")
            return None, False


        # Define the formula for the GLMM
        formula = Formula(f'{target_variable} ~ Perspective * Scene + (1|ID)')

        # print(f"Running GLMM on dataframe with {r_dataframe.nrow} rows and {r_dataframe.ncol} columns")

        try:
            # Fit the GLMM using lme4's glmer function
            gamma_family = r('Gamma(link="inverse")')
            model = lme4.glmer(formula, data=r_dataframe, family=gamma_family)
            summary = r.summary(model)
            # print(summary)

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
    

    def test_IPQ(self, dataframe, target_variable, p_value_index):
        """
        A function to run an Aligned Rank Transform model and return the results.

        Parameters:
        - dataframe (pandas.DataFrame): The dataframe to be analyzed.
        - target_variable (str): The target variable for analysis (e.g., 'Mean_IPQ').
        - p_value_index (int): The index of the p-value to use.

        Returns:
        - tuple: The selected p-value, effect size (generalized eta-squared), and a significance flag.
        """
        artool = importr('ARTool')  # Import ARTool package

        with localconverter(R.default_converter + pandas2ri.converter):
            # Convert Python DataFrame to R DataFrame
            r_dataframe = R.conversion.py2rpy(dataframe)

        # Ensure categorical variables are factors
        try:
            # Convert Perspective and Scene to factors in R
            r_dataframe = r('''
            function(df) {
                df$Perspective <- as.factor(df$Perspective)
                df$Scene <- as.factor(df$Scene)
                return(df)
            }
            ''')(r_dataframe)
        except Exception as e:
            print(f"Error in converting variables to factors: {e}")
            return None, False

        # Define the formula for the ART model
        formula = Formula(f'{target_variable} ~ Perspective * Scene + (1|ID)')

        try:
            # Fit the ART model
            art_model = artool.art(formula, data=r_dataframe)
            
            # Perform ANOVA on the ART model
            anova_results = artool.anova_art(art_model)

            # Convert ANOVA results (R dataframe) to pandas DataFrame
            with localconverter(R.default_converter + pandas2ri.converter):
                anova_df = R.conversion.rpy2py(anova_results)

            # Extract p-values from ANOVA results
            if 'Pr(>F)' in anova_df.columns:
                p_values = anova_df['Pr(>F)'].values
            else:
                raise KeyError("Expected 'Pr(>F)' column not found in ANOVA results")

            # Select the specified p-value
            selected_p_value = p_values[p_value_index]

            # # Extract SS (Sum of Squares) values
            # ss_effect = anova_df.loc[p_value_index, 'Sum Sq']
            # ss_error = anova_df['Sum Sq'].iloc[-1]  # Assuming last row contains residuals
            # ss_total = anova_df['Sum Sq'].sum()

            # # Calculate generalized eta-squared (η²G)
            # effect_size = ss_effect / (ss_total)

            # Check if the selected p-value is significant
            significant = selected_p_value < 0.05

            return selected_p_value, significant

        except Exception as e:
            print(f"Error in ART fitting: {e}")
            return None, None, False





    def sampling(self, dataframe, nb):
        """
        Function to perform sampling based on unique IDs.
        Parameters:
        - dataframe: pandas.DataFrame, the dataset to be sampled.
        - nb: int, the number of unique IDs to sample.

        Returns:
        - pandas.DataFrame, the sampled data containing all rows related to the sampled IDs.
        """
        try:
            # Randomly sample unique IDs
            sampled_ids = dataframe['ID'].drop_duplicates().sample(n=nb, replace=False)
            # print(f"Number of sampled IDs: {len(sampled_ids)}")
            # print(f"Number of unique sampled IDs: {len(set(sampled_ids))}")

            
            # Extract all rows related to the sampled IDs
            df_sampled = dataframe[dataframe['ID'].isin(sampled_ids)]
            # print(f"Sampled DataFrame Dimensions: {df_sampled.shape[0]} rows, {df_sampled.shape[1]} columns")
            # print(f"Sampled IDs: {sampled_ids.tolist()}")
            # for id_ in sampled_ids:
            #     count = len(dataframe[dataframe['ID'] == id_])
            #     print(f"ID: {id_}, Rows in dataframe: {count}")


            # Group by ID and count rows for each ID
            id_counts = df_sampled.groupby('ID').size()

            # Display each ID and the number of rows associated with it
            # print("ID and the number of rows associated:")
            # for id_, count in id_counts.items():
            #     print(f"ID: {id_}, Count: {count}")
            # print(f"Unique IDs in dataframe: {dataframe['ID'].nunique()}")


            return df_sampled
        except Exception as e:
            print(f"Error during sampling: {e}")
            return None


    def prepare_paired_data(self, dataframe, index_value, columns_value):
        """
        Prepare paired data for Wilcoxon test.
        Converts the dataframe to a format where FirstPerson and ThirdPerson data are paired by ID.

        Parameters:
        - dataframe: pandas.DataFrame, the dataset to be prepared.

        Returns:
        - pandas.DataFrame, paired data with columns for FirstPerson and ThirdPerson.
        """
        try:
            # Pivot the data to create pairs
            paired_df = dataframe.pivot(index=["ID",index_value], columns=columns_value, values="Awe_S").reset_index()
            return paired_df
        except Exception as e:
            print(f"Error in preparing paired data: {e}")
            return None

    def wilcoxon_effect_size(self,paired_df):
        """
        Calculate Wilcoxon test and effect size.

        Parameters:
        - paired_df: pandas.DataFrame, paired data with FirstPerson and ThirdPerson columns.

        Returns:
        - tuple: (z-value, p-value, effect size)
        """
        try:
            # Perform Wilcoxon signed-rank test
            z, p_value = sp.stats.wilcoxon(paired_df['FirstPerson'], paired_df['ThirdPerson'])

            # Calculate effect size (r)
            effect_size = z / np.sqrt(len(paired_df))

            return z, p_value, effect_size
        except Exception as e:
            print(f"Error in Wilcoxon test: {e}")
            return None, None, None



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

    def reshape(self, dataframe,target_variable,independent_variable):
        df2=dataframe.pivot(index='ID', columns=independent_variable, values=target_variable)
        return(df2)

    def reshape_long(self, dataframe):
        dataframe["id"] = dataframe.index
        df_long = pd.melt(dataframe, id_vars=['id'], value_vars=['Asynchronous', 'Synchronous'])
        return(df_long)

    def writing(self, file_name, p_value, conformity, ef_size, p_value_ipq, conf_emb_ipq):
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.nb_exp, int(self.nb_parti), p_value, conformity, ef_size, p_value_ipq, conf_emb_ipq])

    def wilcox_ef_size(self, dataframe):
        ef_size = rstatix.wilcox_effsize(data = dataframe, formula = Formula('value ~ Condition'), paired = True)
        return(ef_size)
