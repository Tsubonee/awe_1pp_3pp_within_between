import pandas as pd
import matplotlib.pyplot as plt

# CSVファイルのパスを指定
file_path = "Pourcentage_Conforme_within.csv"

column_names = ["x", "col2", "y_within_3", "col4", "col5", "col6", "y_within_7"]

# ファイルを読み込む
try:
    df_within = pd.read_csv(file_path, header=None)
    df_within.columns = column_names
    print("CSVファイルを正常に読み込みました。")
    print(df_within.head())  # データの最初の5行を表示
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {file_path}")
except pd.errors.EmptyDataError:
    print("エラー: CSVファイルが空です。")

file_path = "Pourcentage_Conforme_between.csv"

# ファイルを読み込む
try:
    df_between = pd.read_csv(file_path, header=None)
    df_between.columns = column_names
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

def plot_graph(x, y1, y2, title, filename, ylabel, xlabel):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker='o', label="within")
    plt.plot(x, y2, marker='o', label="between")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-5, 105)  # y軸を0から100に設定
    plt.xlim(5,45)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename, format='pdf')  # グラフをPDFで保存
    plt.show()

plot_graph(x_within, y_within_3, y_between_3, "Awe-S Score(Perspective)", "Awe_Perspective.pdf", "Awe-S Score Consistency [%]", "Participant number")

plot_graph(x_between, y_within_7, y_between_7, "IPQ Score(Perspective)", "IPQ_Perspective.pdf", "IPQ Score Consistency [%]", "Participant number")

# Scene plot

# CSVファイルのパスを指定
file_path = "Pourcentage_Conforme_within_Scene.csv"

# ファイルを読み込む
try:
    df_within = pd.read_csv(file_path, header=None)
    df_within.columns = column_names
    print("CSVファイルを正常に読み込みました。")
    print(df_within.head())  # データの最初の5行を表示
except FileNotFoundError:
    print(f"エラー: ファイルが見つかりません: {file_path}")
except pd.errors.EmptyDataError:
    print("エラー: CSVファイルが空です。")

file_path = "Pourcentage_Conforme_between_Scene.csv"

# ファイルを読み込む
try:
    df_between = pd.read_csv(file_path, header=None)
    df_between.columns = column_names
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

def plot_graph(x, y1, y2, title, filename, ylabel, xlabel):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, marker='o', label="within")
    plt.plot(x, y2, marker='o', label="between")
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-5, 105)  # y軸を0から100に設定
    plt.xlim(5,45)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.savefig(filename, format='pdf')  # グラフをPDFで保存
    plt.show()

plot_graph(x_within, y_within_3, y_between_3, "Awe-S Score(Scene)", "Awe_Scene.pdf", "Awe-S Score Consistency [%]", "Participant number")

plot_graph(x_between, y_within_7, y_between_7, "IPQ Score(Scene)", "IPQ_Scene.pdf", "IPQ Score Consistency [%]", "Participant number")


