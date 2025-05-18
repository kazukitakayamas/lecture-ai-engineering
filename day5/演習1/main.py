import os
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import pickle
import optuna
from optuna.samplers import TPESampler  # Tree-structured Parzen Estimator (TPE)によるベイズ探索
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature


# データ準備
def prepare_data(test_size=0.2, random_state=42):
    # Titanicデータセットの読み込み
    path = "data/Titanic.csv"
    data = pd.read_csv(path)

    # 必要な特徴量の選択と前処理
    data = data[["Pclass", "Sex", "Age", "Fare", "Survived"]].dropna()
    data["Sex"] = LabelEncoder().fit_transform(data["Sex"])  # 性別を数値に変換

    # 整数型の列を浮動小数点型に変換
    data["Pclass"] = data["Pclass"].astype(float)
    data["Sex"] = data["Sex"].astype(float)
    data["Age"] = data["Age"].astype(float)
    data["Fare"] = data["Fare"].astype(float)
    data["Survived"] = data["Survived"].astype(float)

    X = data[["Pclass", "Sex", "Age", "Fare"]]
    y = data["Survived"]

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# Optunaの目的関数
def objective(trial, X_train, y_train):
    # ハイパーパラメータの候補をOptunaが提案
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42
    }
    
    # RandomForestClassifierのインスタンス化
    model = RandomForestClassifier(**params)
    
    # 交差検証によるスコアの評価
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    return score.mean()


# 最適なモデルを学習
def train_best_model(best_params, X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(**best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


# モデルをMLflowに記録
def log_model(model, accuracy, params):
    with mlflow.start_run():
        # パラメータをログ
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # メトリクスをログ
        mlflow.log_metric("accuracy", accuracy)

        # モデルのシグネチャを推論
        signature = infer_signature(X_train, model.predict(X_train))

        # モデルを保存
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            input_example=X_test.iloc[:5],  # 入力例を指定
        )
        # accuracyとparamsを改行して表示
        print(f"モデルのログ記録値 \naccuracy: {accuracy}\nparams: {params}")


# メイン処理
if __name__ == "__main__":
    # データ準備（固定シードを使用）
    random_state = 42
    test_size = 0.2
    X_train, X_test, y_train, y_test = prepare_data(
        test_size=test_size, random_state=random_state
    )
    
    # Optunaによるハイパーパラメータ最適化の実行
    print("Optunaによるハイパーパラメータ最適化を開始します...")
    study = optuna.create_study(
        direction="maximize",  # 精度を最大化
        sampler=TPESampler(seed=random_state)  # ベイズ探索（TPEサンプラー）を使用
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=50  # 試行回数（必要に応じて調整可能）
    )
    
    # 最適なハイパーパラメータを取得
    best_params = study.best_params
    best_params['random_state'] = random_state  # ランダムシードを追加
    print(f"最適なハイパーパラメータ: {best_params}")
    print(f"最高精度 (交差検証): {study.best_value:.4f}")
    
    # 最適なハイパーパラメータでモデルを再学習
    model, accuracy = train_best_model(best_params, X_train, X_test, y_train, y_test)
    print(f"テストデータでの精度: {accuracy:.4f}")
    
    # MLflowでモデルとパラメータを記録
    params = {
        "test_size": test_size,
        "data_random_state": random_state,
        **best_params
    }
    log_model(model, accuracy, params)
    
    # モデルをファイルに保存
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"titanic_model_optuna.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"モデルを {model_path} に保存しました")
    
    # Optunaの最適化履歴をプロット
    try:
        import matplotlib.pyplot as plt
        from optuna.visualization import plot_optimization_history, plot_param_importances

        # 最適化履歴のプロット
        fig = plot_optimization_history(study)
        fig.write_image("optuna_optimization_history.png")
        
        # パラメータ重要度のプロット
        param_importance = plot_param_importances(study)
        param_importance.write_image("optuna_param_importances.png")
        
        print("Optunaの可視化結果を保存しました")
    except ImportError:
        print("可視化ライブラリがインストールされていないため、プロットはスキップされました")
        print("プロットを生成するには、plotly と matplotlib をインストールしてください")
