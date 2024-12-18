import argparse
from os import stat
import pandas as pd
import numpy as np

from rpy2.robjects import conversion

from tqdm import tqdm
import scipy as sp

import statsmodels.api as sm
import statsmodels.formula.api as smf

import os

## Get methods from ##
from Stats_test_class import *

parser = argparse.ArgumentParser(description='Between')
parser.add_argument('-p', help = 'participant number - exp number', nargs=2)
args = parser.parse_args() 
file_path = '../Questionnaire/Questionnaire_result_between.csv'

# ファイルが存在するか確認して読み込み
if os.path.exists(file_path):
    df_between= pd.read_csv(file_path)

start = int(args.p[0]) #10
exp = int(args.p[1]) #1000


participant_counts = list(range(start, 41, 5))  # 10 - 41
if 41 not in participant_counts:
    participant_counts.append(41)  # 41を最後に追加
print(f"Participant counts to process: {participant_counts}") 

# 実験処理の実行
for parti in participant_counts: #10 - 41
    stats = Statistics(parti, exp, df_between)
    stats.__init__(parti, exp, df_between)

    for i in tqdm(range(1, exp+10), desc=f"Experiment Progress (Participants: {parti})"):
        try:
            # サンプリング
            df_simu = stats.sampling(df_between, nb=parti).copy()
            df_simu['Scene'] = pd.Categorical(
                df_simu['Scene'],
                categories=['Corridor', 'Snow'],  
                ordered=True
            )

            # GLMM
            p_emb, conf_emb = stats.test_GLMM(df_simu, target_variable='Awe_S', p_value_index=1)

            # paired_df = stats.prepare_paired_data(df_simu, columns_value='Perspective', index_value='Scene')

            # Wilcoxon効果量
            if df_simu is not None:
                z, p_value, effect_size = stats.wilcoxon_effect_size(df_simu)
                print(f"Wilcoxon Test Results for Experiment {i} (Participants: {parti}): z={z}, p_value={p_emb}, effect_size={effect_size}")
            else:  
                print(f"Experiment {i} (Participants: {parti}): Failed to prepare paired data for Wilcoxon test.")
                continue

            p_emb_ipq, conf_emb_ipq = stats.test_IPQ(df_simu, target_variable='Mean_IPQ', p_value_index=1)

            # 結果の書き込み
            stats.writing(str(parti) + '_between_data_descriptions_' + str(exp) + '.csv', p_emb, conf_emb, effect_size, p_emb_ipq, conf_emb_ipq)

        except Exception as e:
            print(f"Error in experiment {i} (Participants: {parti}): {e}")



    # 集計処理
    name = str(parti) + '_between_data_descriptions_'  + str(exp) + '.csv'
    df_count = pd.read_csv(name, names=["nb_simu", "nb_parti", "p_value", "confo", "effect_size", "p_value_ipq", "confo_ipq"])

    ef_size_moy = df_count.effect_size.mean()
    fifth_perc = np.percentile(df_count['effect_size'], 5)
    ninety_fifth_perc = np.percentile(df_count['effect_size'], 95)

    count = 0
    for i in range(0, exp - 1): 
        if (df_count.iat[i, 3] == True):
            count += 1

    pourcent = count * 100 / exp


    count_ipq = 0
    for i in range(0, exp - 1):
        if (df_count.iat[i, 6] == True):
            count_ipq += 1

    pourcent_ipq = count_ipq * 100 / exp

    file_name = 'Pourcentage_Conforme_between.csv'

    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parti, len(df_between), pourcent, ef_size_moy, fifth_perc, ninety_fifth_perc, pourcent_ipq])

# Compare to Scene
for parti in participant_counts:
    stats = Statistics(parti, exp, df_between)
    stats.__init__(parti, exp, df_between)

    for i in tqdm(range(1, exp), desc=f"Experiment Progress (Participants: {parti})"):
        try:
            df_simu = stats.sampling(df_between, nb=parti).copy()

            df_simu['Perspective'] = pd.Categorical(
                df_simu['Perspective'],
                categories=['FirstPerson', 'ThirdPerson'],  
                ordered=True
            )

            # GLMM
            p_emb, conf_emb = stats.test_GLMM(df_simu, target_variable='Awe_S', p_value_index=2)

            paired_df = stats.prepare_paired_data(df_simu, columns_value='Scene', index_value='Perspective')

            # Wilcoxon効果量
            if paired_df is not None:
                z, p_value, effect_size = stats.wilcoxon_effect_size(paired_df)
                print(f"Wilcoxon Test Results for Experiment {i} (Participants: {parti}): z={z}, p_value={p_emb}, effect_size={effect_size}")
            else:  
                print(f"Experiment {i} (Participants: {parti}): Failed to prepare paired data for Wilcoxon test.")
                continue

            p_emb_ipq, conf_emb_ipq = stats.test_IPQ(df_simu, target_variable='Mean_IPQ', p_value_index=2)

            # 結果の書き込み
            stats.writing(str(parti) + '_between_data_descriptions_Scene_' + str(exp) + '.csv', p_emb, conf_emb, effect_size, p_emb_ipq, conf_emb_ipq)

        except Exception as e:
            print(f"Error in experiment {i} (Participants: {parti}): {e}")

    # 集計処理
    name = str(parti) + '_between_data_descriptions_Scene_'  + str(exp) + '.csv'
    df_count = pd.read_csv(name, names=["nb_simu", "nb_parti", "p_value", "confo", "effect_size", "p_value_ipq", "confo_ipq"])

    ef_size_moy = df_count.effect_size.mean()
    fifth_perc = np.percentile(df_count['effect_size'], 5)
    ninety_fifth_perc = np.percentile(df_count['effect_size'], 95)

    count = 0
    for i in range(0, exp - 1):
        if (df_count.iat[i, 3] == True):
            count += 1

    pourcent = count * 100 / exp


    count_ipq = 0
    for i in range(0, exp - 1):
        if (df_count.iat[i, 6] == True):
            count_ipq += 1

    pourcent_ipq = count_ipq * 100 / exp

    file_name = 'Pourcentage_Conforme_between_Scene.csv'

    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parti, len(df_between), pourcent, ef_size_moy, fifth_perc, ninety_fifth_perc, pourcent_ipq])