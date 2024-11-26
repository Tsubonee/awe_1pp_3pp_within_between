import argparse
from os import stat
import pandas as pd
import numpy as np

from rpy2.robjects import conversion


import scipy as sp

import statsmodels.api as sm
import statsmodels.formula.api as smf

np.random.seed(3516)



from Stats_test_class import *

parser = argparse.ArgumentParser(description='Within')
parser.add_argument('-p', help = 'participant number - exp number', nargs=2)
args = parser.parse_args() 

#On importe les données
df_within = pd.read_csv('data_to_publish.csv')

exp = int(args.p[1])

parti = int(args.p[0])

stats = Statistics(parti, exp, df_within)
stats.__init__(parti, exp, df_within)

for i in range(1, exp):
    df_simu = stats.sampling(stats.reshape(df_within), parti) #get sampling data
    p_emb, conf_emb = stats.testing(df_simu)
    with localconverter(R.default_converter + pandas2ri.converter):
            df_r= R.conversion.py2rpy(stats.reshape_long(df_simu))
    ef_size = stats.wilcox_ef_size(df_r)
    with localconverter(R.default_converter + pandas2ri.converter):
        ef_size_pd = R.conversion.rpy2py(ef_size)
    stats.writing(str(stats.nb_parti) + '_within_data_descriptions_' + str(stats.nb_exp) + '.csv', p_emb, conf_emb, ef_size_pd.iloc[0,3])


## For within simulation ##
# name = str(parti) + '_within_data_descriptions' + '.csv'
name = str(parti) + '_within_data_descriptions_' + str(stats.nb_exp) + '.csv'

## Read simulation log for a particular condition(within/between/2nd between) for a specific number of participants ##
df_count = pd.read_csv(name, names=["nb_simu", "nb_parti", "p_value", "confo", "effect_size"])

ef_size_moy = df_count.effect_size.mean()
fifth_perc = np.percentile(df_count['effect_size'], 5)
ninety_fifth_perc = np.percentile(df_count['effect_size'], 95)

count = 0

for i in range(0, exp-1):
    if (df_count.iat[i, 3] == True):
        count += 1

pourcent = count * 100 / exp

file_name = 'Pourcentage_Conforme_within.csv'

with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([exp, parti, pourcent, ef_size_moy, fifth_perc, ninety_fifth_perc])