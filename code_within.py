import argparse
from os import stat
import pandas as pd
import numpy as np

from rpy2.robjects import conversion

from tqdm import tqdm
import scipy as sp

import statsmodels.api as sm
import statsmodels.formula.api as smf

from Stats_test_class import *

# 引数の設定
parser = argparse.ArgumentParser(description='Within')
parser.add_argument('-p', help='participant start and increment', nargs=2)
args = parser.parse_args()

# データの読み込み
df_within = pd.read_csv('Questionnaire_result.csv')

start = int(args.p[0])  
exp = int(args.p[1])  

# 処理を行う参加者数リストを作成
participant_counts = list(range(start, 41, 5))  # 5刻みで41を超えない値まで
if 41 not in participant_counts:
    participant_counts.append(41)  # 41を最後に追加
print(f"Participant counts to process: {participant_counts}") 

# 実験処理の実行
for parti in participant_counts:
    stats = Statistics(parti, exp, df_within)
    stats.__init__(parti, exp, df_within)

    # 実験を実行
    for i in tqdm(range(1, exp), desc=f"Experiment Progress (Participants: {parti})"):
        try:
            # サンプリング
            df_simu = stats.sampling(df_within, nb=parti).copy()

            df_simu['Scene'] = pd.Categorical(
                df_simu['Scene'],
                categories=['Corridor', 'Snow'],  
                ordered=True
            )

            # GLMM
            p_emb, conf_emb = stats.test_GLMM(df_simu, target_variable='Awe_S', p_value_index=1)

            # Wilcoxon
            paired_df = stats.prepare_paired_data(df_simu, columns_value='Perspective', index_value='Scene')

            # Wilcoxon効果量
            if paired_df is not None:
                z, p_value, effect_size = stats.wilcoxon_effect_size(paired_df)
                print(f"Wilcoxon Test Results for Experiment {i} (Participants: {parti}): z={z}, p_value={p_emb}, effect_size={effect_size}")
            else:  
                print(f"Experiment {i} (Participants: {parti}): Failed to prepare paired data for Wilcoxon test.")
                continue

            # 結果の書き込み
            stats.writing(str(parti) + '_within_data_descriptions_' + str(exp) + '.csv', p_emb, conf_emb, effect_size)

        except Exception as e:
            print(f"Error in experiment {i} (Participants: {parti}): {e}")

    # 集計処理
    name = str(parti) + '_within_data_descriptions_'  + str(exp) + '.csv'
    df_count = pd.read_csv(name, names=["nb_simu", "nb_parti", "p_value", "confo", "effect_size"])

    ef_size_moy = df_count.effect_size.mean()
    fifth_perc = np.percentile(df_count['effect_size'], 5)
    ninety_fifth_perc = np.percentile(df_count['effect_size'], 95)

    count = 0
    for i in range(0, exp - 1):
        if (df_count.iat[i, 3] == True):
            count += 1

    pourcent = count * 100 / exp

    file_name = 'Pourcentage_Conforme_within.csv'

    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([parti, len(df_within), pourcent, ef_size_moy, fifth_perc, ninety_fifth_perc])
