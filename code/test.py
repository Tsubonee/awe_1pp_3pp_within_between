import pandas as pd
from Stats_test_class import Statistics

# テスト用データ
data = {
    'ID': [1, 1, 2, 2],
    'Perspective': ['First', 'Third', 'First', 'Third'],
    'Scene': ['SceneA', 'SceneA', 'SceneB', 'SceneB'],
    'Awe_S': [0.85, 0.70, 0.90, 0.65],
}
df = pd.DataFrame(data)

# Statisticsクラスのインスタンスを作成
stats = Statistics(nb_participant=2, nb_exp=1, dataframe=df)

# GLMMを実行
p_values, significant = stats.test_GLMM(dataframe=df, target_variable='Awe_S')
print(f"P-values: {p_values}, Significant: {significant}")
