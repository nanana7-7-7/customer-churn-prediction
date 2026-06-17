# Customer Churn Prediction

## Overview

This project builds a machine learning model to predict customer churn using telco customer data.

通信会社の顧客データを用いて、顧客が解約するかどうかを予測する機械学習モデルを構築します。  
EDA、前処理、モデル比較、評価、特徴量重要度の可視化、Streamlitアプリ化までを行うエンドツーエンドの機械学習プロジェクトです。

## Background

サブスクリプション型サービスや通信サービスでは、既存顧客の解約を事前に予測することが重要です。  
解約リスクの高い顧客を早期に発見できれば、キャンペーン、サポート強化、契約プランの提案などのリテンション施策につなげることができます。

本プロジェクトでは、顧客属性、契約情報、利用サービス情報から、顧客の解約リスクを予測します。

## Dataset

This project uses the IBM Telco Customer Churn dataset.

IBMのTelco Customer Churnサンプルデータは、架空の通信会社の顧客解約を扱うデータで、性別、扶養家族、月額料金、契約・サービス情報などを含みます。解約列は顧客が離脱したかどうかを示します。【1-6ced32】【2-4d2fd2】

Kaggle上のIBM Telco customer churn datasetは、架空の通信会社が提供する電話・インターネットサービスの顧客データで、7,043件の観測データと顧客属性・契約情報・解約ラベルを含むデータセットです。【3-583d18】

## Problem Definition

- Task: Binary Classification
- Target: Churn
- Goal: Predict whether a customer will churn or not

## Tools and Libraries

- Python
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- Streamlit
- joblib

## Project Structure

```text
customer-churn-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv
│   └── processed/
│       └── processed_churn.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
├── models/
│   └── churn_model.pkl
├── reports/
│   ├── figures/
│   │   ├── churn_distribution.png
│   │   ├── confusion_matrix.png
│   │   └── feature_importance.png
│   └── model_report.md
└── docs/
    └── project_summary.md
