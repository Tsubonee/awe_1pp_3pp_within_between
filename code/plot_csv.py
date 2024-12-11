import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパスを指定
file_path = "Pourcentage_Conforme_within.csv"

# ファイルを読み込む
try:
    df_within = pd.read_csv(file_path)
    print("CSVファイルを正常に読み込みました。")
    print(df_within.head())  # データの最初の5行を表示
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {file_path}")
except pd.errors.EmptyDataError:
    print("エラー: CSVファイルが空です。")

file_path = "Pourcentage_Conforme_between.csv"

# ファイルを読み込む
try:
    df_between = pd.read_csv(file_path)
    print("CSVファイルを正常に読み込みました。")
    print(df_between.head())  # データの最初の5行を表示
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {file_path}")
except pd.errors.EmptyDataError:
    print("エラー: CSVファイルが空です。")

x_within = df_within.iloc[:, 0]
y_within_3 = df_within.iloc[:, 2]
y_within_7 = df_within.iloc[:, 6]

x_between = df_between.iloc[:, 0]
y_between_3 = df_between.iloc[:, 2]
y_between_7 = df_between.iloc[:, 6]

def plot_graph(x, y1, y2, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker='o', label="3rd column")
    plt.plot(x, y2, marker='o', label="7th column")
    plt.xlabel("Awe-S Score")
    plt.ylabel("Participant number")
    plt.title(title)
    plt.legend()
    plt.grid()
    # plt.savefig(filename)  # グラフを保存
    plt.show()

plot_graph(x_within, y_within_3, y_between_3, "Awe-S Score(Perspective)", "Awe_Perspective.png")

plot_graph(x_between, y_within_7, y_between_7, "IPQ Score(Perspective)", "IPQ_Perspective.png")


