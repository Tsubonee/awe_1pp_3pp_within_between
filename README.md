# awe_1pp_3pp_within_between

このリポジトリには、以下の論文で使用された実験データを基にした追加解析用のコードが含まれています。

**First-Person Perspective Induces Stronger Feelings of Awe and Presence Compared to Third-Person Perspective in Virtual Reality**  
[論文リンクはこちら](https://dl.acm.org/doi/10.1145/3678957.3685753)

論文で使用した主要なコードとデータセットについては、以下のOSFプロジェクトで公開しています：  
[OSFプロジェクトリンク](https://osf.io/eq9vp/)

このリポジトリには、OSFで公開されているコードを基にした追加解析スクリプトと実験データを収録しています。

---

## ファイル構成

### データ
- **`Questionnaire_result.csv`**  
  実験で収集されたデータを格納したCSVファイルです。解析の基礎データとして使用します。

### スクリプト
- **`code_within.py`**  
  解析用のメインスクリプトです。  
  サンプルサイズや仮想実験の試行回数を指定して解析を実行できます。

---

## 解析方法

以下のコマンドを使用して解析を実行してください：

```python
python3 code_within.py -p <サンプルサイズの増加ステップ> <仮想実験の試行回数>
```

### 実行例

以下のコマンドは、元データセット（41人の参加者）の中から10人から41人まで5人刻みでサンプリングし、1,000回の仮想実験を実行します：

```python
python3 code_within.py -p 10 1000
```

### 必要環境
Python 3 がインストールされている環境で動作します。
必要なPythonパッケージが不足している場合は、以下のコマンドでインストールしてください：

```python
pip install -r requirements.txt
```
