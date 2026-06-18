# Customer Churn Prediction
 
## 概要

このプロジェクトは、通信会社の顧客データを用いて、顧客が解約するかどうかを予測する機械学習プロジェクトです。

探索的データ分析、前処理、モデル学習、評価、予測スクリプト作成、Streamlitアプリ化までを一通り実装しています。

## 背景

サブスクリプション型サービスや通信サービスでは、既存顧客の解約を事前に予測することが重要です。  
解約リスクの高い顧客を早期に把握できれば、キャンペーン、サポート強化、契約プランの提案などのリテンション施策につなげることができます。

本プロジェクトでは、顧客属性、契約情報、利用サービス情報をもとに、顧客の解約リスクを予測します。

## 使用データ

本プロジェクトでは、Kaggle の Telco Customer Churn データセットを使用しています。

このデータセットには、架空の通信会社における以下のような顧客情報が含まれています。

- 顧客属性
- 契約情報
- 利用サービス情報
- 月額料金
- 累計料金
- 解約有無

### 目的変数

```text
Churn
```

- Yes：解約した顧客
- No：解約していない顧客

## 問題設定

本プロジェクトは、二値分類問題です。

```text
入力：顧客情報
出力：解約 / 非解約
```

顧客が解約する可能性があるかどうかを予測することを目的としています。

## 使用技術・ライブラリ

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib
- Streamlit

## ディレクトリ構成

```text
Customer Churn Prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── raw/
│       └── Telco-Customer-Churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
├── models/
│   └── churn_model.pkl
└── reports/
    ├── model_results.csv
    ├── evaluation_report.txt
    └── figures/
        ├── confusion_matrix.png
        ├── feature_importance.png
        └── roc_curve.png
```

## 作業の流れ

- データセットの読み込み
- 探索的データ分析
- データ前処理
- 学習データとテストデータへの分割
- モデル学習
- モデル比較
- モデル評価
- 学習済みモデルの保存
- 予測スクリプトの作成
- Streamlitアプリの作成

## 探索的データ分析

`notebooks/01_eda.ipynb` では、以下を確認しました。

- データの基本構造
- 欠損値・重複の確認
- 目的変数 `Churn` の分布
- 数値変数と解約の関係
- カテゴリ変数と解約の関係
- 解約に影響しそうな特徴量の仮説

### 分析から得られた主な気づき

EDAから、以下の傾向が見られました。

- 解約していない顧客の方が多く、目的変数には偏りがある
- `TotalCharges` は文字列型として読み込まれていた
- `TotalCharges` を数値型に変換すると、11件の欠損値が確認された
- `TotalCharges` の欠損は `tenure = 0` の顧客に発生していた
- `tenure` が短い顧客ほど解約しやすい傾向がある
- Month-to-month 契約の顧客は解約率が高い傾向がある
- `MonthlyCharges` が高い顧客は解約しやすい可能性がある
- TechSupport や OnlineSecurity がない顧客は解約率が高い傾向がある
- Electronic check を利用している顧客は解約率が高い傾向がある

## 前処理

以下の前処理を行いました。

- `TotalCharges` を数値型に変換
- `TotalCharges` の欠損値を 0 で補完
- `customerID` を削除
- `Churn` を `No=0`, `Yes=1` に変換
- 数値変数に `StandardScaler` を適用
- カテゴリ変数に `OneHotEncoder` を適用

## 使用モデル

以下の3つのモデルを比較しました。

- Logistic Regression 
- Random Forest 
- Gradient Boosting 

## 評価指標

以下の評価指標を使用しました。

- 正解率（Accuracy）
- 適合率（Precision）
- 再現率（Recall）
- F1スコア
- ROC-AUC
- 混同行列
- ROC曲線

解約予測では、単純な正解率だけではなく、実際に解約する顧客をどれだけ検出できるかが重要です。  
そのため、再現率、F1スコア、ROC-AUCも確認しました。

## モデル結果

モデル比較の結果は、以下のファイルに保存しています。

```text
reports/model_results.csv
```

### 結果概要

| モデル | 正解率 | 適合率 | 再現率 | F1スコア | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8055 | 0.6572 | 0.5588 | 0.6040 | 0.8421 |
| RandomForest | 0.7864 | 0.6254 | 0.4866 | 0.5474 | 0.8185 |
| Gradient Boosting | 0.8027 | 0.6655 | 0.5160 | 0.5813 | 0.8433 |

## 評価レポート

評価レポートは、以下に保存しています。

```text
reports/evaluation_report.txt
```

## 可視化

### 混同行列

<img width="1800" height="1500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/46463b2a-e766-42e2-942e-af571296caac" />

### ROC曲線

<img width="1800" height="1500" alt="roc_curve" src="https://github.com/user-attachments/assets/c00ab3ef-c069-4938-87fd-67fef6cb0a65" />

## Streamlitアプリ

本プロジェクトでは、顧客の解約リスクを予測するStreamlitアプリも作成しています。

アプリでは、顧客情報を入力すると以下を確認できます。

- 解約確率
- リスクレベル
- 予測結果

### リスクレベルの定義

```text
高リスク：解約確率 70%以上
中リスク：解約確率 40%以上 70%未満
低リスク：解約確率 40%未満
```

## 実行方法

### 1. リポジトリをクローン

```bash
git clone https://github.com/nanana7-7-7/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. 仮想環境の作成

Windowsの場合：

```bash
python -m venv .venv
.venv\Scripts\activate
```

Mac / Linuxの場合：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 4. データセットの準備

Telco Customer Churn データセットをダウンロードし、以下に配置します。

```text
data/raw/Telco-Customer-Churn.csv
```

### 5. モデルの学習

```bash
python src/train.py
```

このスクリプトでは複数モデルを学習し、最も性能の良いモデルを以下に保存します。

```text
models/churn_model.pkl
```

### 6. モデルの評価

```bash
python src/evaluate.py
```

このスクリプトにより、以下のファイルが生成されます。

```text
reports/evaluation_report.txt
reports/figures/confusion_matrix.png
reports/figures/roc_curve.png
```

### 7. 1人分の顧客データで予測

```bash
python src/predict.py
```

### 8. Streamlitアプリの起動

```bash
streamlit run app/streamlit_app.py
```

## 予測結果の例

```text
予測結果
=================
予測ラベル：1
解約確率：67.32%
リスクレベル：中リスク
判定：この顧客は解約する可能性があります。
```

## ファイル説明

### `notebooks/01_eda.ipynb`

探索的データ分析を行うNotebookです。

### `notebooks/02_modeling.ipynb`

モデル学習・評価を行うNotebookです。

### `src/train.py`

複数モデルを学習し、ROC-AUCが最も高いモデルを保存します。

### `src/evaluate.py`

保存済みモデルを読み込み、評価指標とグラフを出力します。

### `src/predict.py`

1人分の顧客情報から解約確率を予測します。

### `app/streamlit_app.py`

Streamlitによる予測アプリです。

## 学んだこと

このプロジェクトを通して、以下を学びました。

- pandasを用いたデータ確認と前処理
- matplotlib / seabornによる可視化
- scikit-learnのPipeline設計
- One-Hot Encodingと標準化
- 複数モデルの比較
- 正解率以外の評価指標の重要性
- joblibによるモデル保存
- Streamlitによる機械学習アプリ化
- GitHub向けのプロジェクト整理

## 今後の改善案

今後の改善案は以下です。

- ハイパーパラメータチューニング
- 閾値調整による再現率の改善
- SHAPによる予測理由の可視化
- クラス不均衡への対応
- LightGBM / XGBoost の追加
- Streamlit Cloudへのデプロイ
- UIデザインの改善
