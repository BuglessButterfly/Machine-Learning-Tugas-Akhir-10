from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# =============================================================
# FIX: Matplotlib backend NON-GUI (aman di Flask)
# =============================================================
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import io, base64

app = Flask(__name__)

# =============================================================
# UTIL: FIGURE -> BASE64
# =============================================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================
# (ELBOW CV + CONFUSION MATRIX)
# =============================================================
def make_elbow_cm_grid(k_values, errors, cm, classes_, best_k):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---- subplot kiri: Elbow Method (CV error) ----
    axes[0].plot(list(k_values), errors, marker="o")
    axes[0].set_xlabel("Nilai K")
    axes[0].set_ylabel("Error Rate (CV)")
    axes[0].set_title("Elbow Method KNN (K vs Error Rate) - Cross Validation")
    axes[0].grid(True)

    # highlight best_k
    axes[0].axvline(best_k, linestyle="--")
    axes[0].text(best_k, min(errors), f" best k={best_k}", va="bottom")

    # ---- subplot kanan: Confusion Matrix ----
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=classes_, yticklabels=classes_,
        ax=axes[1]
    )
    axes[1].set_title(f"Confusion Matrix (k={best_k})")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    # jarak antar subplot biar tidak nempel
    fig.subplots_adjust(wspace=0.35)

    return fig_to_base64(fig)

# =============================================================
# TRAIN MODEL SAAT APLIKASI START (ELBOW CV SESUAI NOTEBOOK)
# =============================================================
def train_knn_model(csv_path="Iris.csv"):
    # 1) Load data
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # 2) Pisah fitur & target
    target_col = "label"
    if target_col not in df.columns:
        raise ValueError(
            f"Kolom target '{target_col}' tidak ditemukan. Kolom ada: {df.columns.tolist()}"
        )

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # ambil numerik
    X = X.select_dtypes(include="number")

    # pastikan urutan fitur konsisten (Iris)
    feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
    missing = [f for f in feature_names if f not in X.columns]
    if missing:
        raise ValueError(
            f"Kolom fitur kurang: {missing}. Kolom di CSV: {list(X.columns)}"
        )

    X = X[feature_names]
    feature_keys = [f.replace(" ", "_") for f in feature_names]

    # 3) Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4) Scaling (dipakai train final & prediksi)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    # =============================================================
    # 5) ELBOW METHOD (K=1..40) pakai CV (SESUAI PERMINTAAN)
    # =============================================================
    k_values = range(1, 41)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv_scores = []
    errors = []

    for k in k_values:
        # Pipeline biar scaling dilakukan tiap fold (CV valid)
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k))
        ])

        score = cross_val_score(
            pipe, X_train, y_train,
            cv=cv, scoring="accuracy"
        ).mean()

        cv_scores.append(score)
        errors.append(1 - score)

    # best_k dari error minimum CV
    best_k = int(np.argmin(errors) + 1)

    # 6) Train final model pakai best_k pada TRAIN set (scaled)
    final_knn = KNeighborsClassifier(n_neighbors=best_k)
    final_knn.fit(X_train_scaled, y_train)

    # 7) Evaluasi final di TEST set
    y_pred = final_knn.predict(X_test_scaled)
    final_acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classes_ = list(final_knn.classes_)

    # ====== (Elbow CV + CM) ======
    grid_plot = make_elbow_cm_grid(k_values, errors, cm, classes_, best_k)

    # classification report untuk tampil di html
    report_dict = classification_report(y_test, y_pred, output_dict=True)

    return (final_knn, scaler, feature_names, feature_keys,
            best_k, final_acc, classes_, X, y,
            grid_plot, report_dict)


(model, scaler, feature_names, feature_keys,
 best_k, final_acc, classes_, X_full, y_full,
 grid_plot, report_dict) = train_knn_model()


label_to_name = {
    "Iris-setosa": "Iris Setosa",
    "Iris-versicolor": "Iris Versicolor",
    "Iris-virginica": "Iris Virginica"
}

label_to_img_url = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/a/a7/Irissetosa1.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# =============================================================
# SIAPKAN DATASET UNTUK DROPDOWN
# =============================================================
def build_dataset_rows():
    rows = []
    for idx, row in X_full.iterrows():
        label = y_full.loc[idx]
        rows.append({
            "idx": int(idx),
            "label": label,
            "label_name": label_to_name.get(label, label),
            "vals": [float(row[f]) for f in feature_names]
        })
    return rows

dataset_rows = build_dataset_rows()

# Precompute mean per kelas (biar gak hitung ulang tiap predict)
class_means = X_full.groupby(y_full).mean()

# =============================================================
# GRAFIK INPUT vs MEAN SEMUA KELAS (highlight prediksi)
# =============================================================
def make_input_vs_classmean_plot(input_values, pred_label):
    x = np.arange(len(feature_names))
    input_values = np.asarray(input_values, dtype=float)

    fig = plt.figure(figsize=(7, 4))

    for c in classes_:
        class_mean = class_means.loc[c].values

        if c == pred_label:
            plt.plot(x, class_mean, marker="o", linewidth=3,
                     label=f"Mean {c} (Prediksi)")
        else:
            plt.plot(x, class_mean, marker="o", alpha=0.5,
                     label=f"Mean {c}")

    plt.plot(x, input_values, marker="s", linewidth=3, label="Input Anda")

    plt.xticks(x, feature_names, rotation=20)
    plt.title("Perbandingan Input vs Mean Semua Kelas")
    plt.xlabel("Fitur")
    plt.ylabel("Nilai")
    plt.legend()
    plt.grid(True)

    return fig_to_base64(fig)


# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        feature_names=feature_names,
        feature_keys=feature_keys,
        best_k=best_k,
        final_acc=round(final_acc, 4),
        classes_=classes_,
        grid_plot=grid_plot,
        pred_name=None,
        pred_img_url=None,
        input_plot=None,
        input_values=None,
        error=None,
        dataset_rows=dataset_rows,
        report=report_dict
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_values = []
        for key in feature_keys:
            val_str = request.form.get(key)
            if val_str is None or val_str.strip() == "":
                raise ValueError(f"Input untuk '{key}' kosong.")
            input_values.append(float(val_str))

        input_df = pd.DataFrame([input_values], columns=feature_names)
        input_scaled = scaler.transform(input_df)

        pred_label = model.predict(input_scaled)[0]
        pred_name = label_to_name.get(pred_label, pred_label)
        pred_img_url = label_to_img_url.get(pred_label, None)

        input_plot = make_input_vs_classmean_plot(input_values, pred_label)

        return render_template(
            "index.html",
            feature_names=feature_names,
            feature_keys=feature_keys,
            best_k=best_k,
            final_acc=round(final_acc, 4),
            classes_=classes_,
            grid_plot=grid_plot,
            pred_name=pred_name,
            pred_img_url=pred_img_url,
            input_plot=input_plot,
            input_values=input_values,
            error=None,
            dataset_rows=dataset_rows,
            report=report_dict
        )

    except Exception as e:
        return render_template(
            "index.html",
            feature_names=feature_names,
            feature_keys=feature_keys,
            best_k=best_k,
            final_acc=round(final_acc, 4),
            classes_=classes_,
            grid_plot=grid_plot,
            pred_name=None,
            pred_img_url=None,
            input_plot=None,
            input_values=None,
            error=str(e),
            dataset_rows=dataset_rows,
            report=report_dict
        )

if __name__ == "__main__":
    app.run(debug=True)