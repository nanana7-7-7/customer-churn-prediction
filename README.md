# Customer Churn Prediction

## 1. Overview
通信会社の顧客データを用いて、顧客が解約するかどうかを予測する機械学習モデルを構築しました。

## 2. Background
サブスクリプション型サービスや通信サービスでは、既存顧客の解約を事前に予測し、リテンション施策を行うことが重要です。
本プロジェクトでは、顧客属性、契約情報、利用サービス情報をもとに、解約リスクを予測します。

## 3. Dataset
IBM Telco Customer Churn Datasetを使用しました。

## 4. Problem Definition
- Task: Binary Classification
- Target: Churn
- Goal: 顧客が解約する可能性を予測する

## 5. Tools
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Streamlit

## 6. Workflow
1. データ読み込み
2. データ理解
3. 探索的データ分析
4. 前処理
5. モデル学習
6. モデル評価
7. 特徴量重要度の確認
8. Streamlitアプリ化

## 7. Models
- Logistic Regression
- Random Forest
- Gradient Boosting

## 8. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

## 9. Results
最終モデルの評価結果を記載します。

## 10. Key Findings
分析から得られた知見を記載します。

## 11. Streamlit App
アプリの使い方やスクリーンショットを記載します。

## 12. Future Improvements
- SHAPによる説明性の向上
- ハイパーパラメータチューニング
- デプロイ
- 不均衡データへの対応
