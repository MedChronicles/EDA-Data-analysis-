import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml, load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RNG = 42
np.random.seed(RNG)

OUT = "outputs"  
import os
os.makedirs(OUT, exist_ok=True)

def load_from_kaggle_csv(path="train.csv"):
    """Optional loader for the Kaggle 'Digit Recognizer' CSV
    (https://www.kaggle.com/c/digit-recognizer/data), which stores
    one row per image: label, pixel0, pixel1, ..., pixel783."""
    df = pd.read_csv(path)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype(np.float64)
    return X, y, (28, 28)


def load_mnist():
    """Load MNIST. Tries OpenML (full 70k x 784 dataset) first, then
    falls back to the offline UCI digits dataset bundled with sklearn."""
    try:
        print("Attempting to download full MNIST (70,000 x 784) from OpenML...")
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True,
                             as_frame=False, parser="auto")
        y = y.astype(int)
        print("  -> Success: full MNIST loaded from OpenML.")
        return X.astype(np.float64), y, (28, 28), "OpenML mnist_784 (full MNIST)"
    except Exception as e:
        print(f"  -> No internet access to OpenML ({type(e).__name__}). "
              f"Falling back to offline UCI 'Optical Digits' dataset.")
        data = load_digits()
        X = data.data.astype(np.float64)
        y = data.target.astype(int)
        return X, y, (8, 8), "sklearn load_digits (UCI Optical Recognition of Handwritten Digits)"


print("=" * 70)
print("PART A: DATA UNDERSTANDING AND PREPROCESSING")
print("=" * 70)

X_raw, y, IMG_SHAPE, SOURCE = load_mnist()
n_samples, n_features = X_raw.shape
n_classes = len(np.unique(y))
pixel_max = X_raw.max()

print(f"\nData source           : {SOURCE}")
print(f"Number of samples     : {n_samples}")
print(f"Number of classes     : {n_classes}  -> {sorted(np.unique(y).tolist())}")
print(f"Image dimensions      : {IMG_SHAPE[0]} x {IMG_SHAPE[1]} = {n_features} pixels (features)")
print(f"Raw pixel value range : [{X_raw.min():.1f}, {pixel_max:.1f}]")


fig, axes = plt.subplots(2, 5, figsize=(10, 4.5))
rng = np.random.RandomState(RNG)
sample_idx = rng.choice(n_samples, 10, replace=False)
for ax, idx in zip(axes.ravel(), sample_idx):
    ax.imshow(X_raw[idx].reshape(IMG_SHAPE), cmap="gray")
    ax.set_title(f"Label: {y[idx]}", fontsize=10)
    ax.axis("off")
fig.suptitle(f"Sample Handwritten Digits ({SOURCE})", fontsize=12)
plt.tight_layout(h_pad=2.0)
plt.savefig(f"{OUT}/01_sample_digits.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT}/01_sample_digits.png")

X_norm = X_raw / pixel_max
print(f"Normalized pixel range: [{X_norm.min():.2f}, {X_norm.max():.2f}]")


print("\n" + "=" * 70)
print("PART B: PRINCIPAL COMPONENT ANALYSIS (PCA)")
print("=" * 70)

max_components = min(n_samples, n_features)
pca_full = PCA(n_components=max_components, random_state=RNG)
pca_full.fit(X_norm)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

def n_components_for(threshold):
    return int(np.argmax(cumulative >= threshold) + 1)

n90 = n_components_for(0.90)
n95 = n_components_for(0.95)
n99 = n_components_for(0.99)

print(f"\nComponents needed to retain 90% variance: {n90}")
print(f"Components needed to retain 95% variance: {n95}")
print(f"Components needed to retain 99% variance: {n99}")
print(f"(Original feature space: {n_features} dimensions)")

n_show = min(50, max_components)
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_show + 1), explained[:n_show], "o-", linewidth=1.5, markersize=4)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title(f"Scree Plot (first {n_show} components)")
plt.tight_layout()
plt.savefig(f"{OUT}/02_scree_plot.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/02_scree_plot.png")

plt.figure(figsize=(8, 5))
plt.plot(range(1, max_components + 1), cumulative, linewidth=2)
for thresh, n_c, color in [(0.90, n90, "green"), (0.95, n95, "orange"), (0.99, n99, "red")]:
    plt.axhline(thresh, color=color, linestyle="--", linewidth=1,
                label=f"{int(thresh*100)}% @ {n_c} PCs")
    plt.axvline(n_c, color=color, linestyle="--", linewidth=1)
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance vs. Number of Components")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/03_cumulative_variance.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/03_cumulative_variance.png")

pca_2 = PCA(n_components=2, random_state=RNG)
X_pca_2d = pca_2.fit_transform(X_norm)

n_50 = min(50, max_components)
pca_50 = PCA(n_components=n_50, random_state=RNG)
X_pca_50d = pca_50.fit_transform(X_norm)

print(f"\n2D PCA projection variance captured : {pca_2.explained_variance_ratio_.sum()*100:.2f}%")
print(f"{n_50}D PCA projection variance captured: {pca_50.explained_variance_ratio_.sum()*100:.2f}%")

plt.figure(figsize=(8, 7))
scatter = plt.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap="tab10", s=8, alpha=0.6)
plt.colorbar(scatter, label="Digit", ticks=range(10))
plt.xlabel(f"PC1 ({pca_2.explained_variance_ratio_[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({pca_2.explained_variance_ratio_[1]*100:.1f}% var)")
plt.title("2D PCA Projection of Handwritten Digits")
plt.tight_layout()
plt.savefig(f"{OUT}/04_pca_2d_projection.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/04_pca_2d_projection.png")

print("\n" + "=" * 70)
print("PART C: t-SNE VISUALIZATION")
print("=" * 70)

TSNE_N = min(3000, n_samples)
sub_idx = rng.choice(n_samples, TSNE_N, replace=False)
X_sub_raw = X_norm[sub_idx]
X_sub_pca50 = X_pca_50d[sub_idx]
y_sub = y[sub_idx]
print(f"\nUsing a random subsample of {TSNE_N} points for all t-SNE runs "
      f"(t-SNE does not scale well to very large N).")

def run_tsne(X, perplexity, label):
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca",
                learning_rate="auto", random_state=RNG, max_iter=1000)
    emb = tsne.fit_transform(X)
    dt = time.time() - t0
    print(f"  t-SNE [{label}, perplexity={perplexity}] done in {dt:.1f}s")
    return emb, dt

emb_raw, t_raw = run_tsne(X_sub_raw, 30, "raw pixels")
emb_pca, t_pca = run_tsne(X_sub_pca50, 30, "PCA-50")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for ax, emb, title, dt in [
    (axes[0], emb_raw, f"t-SNE on Raw {n_features}-D Pixel Data", t_raw),
    (axes[1], emb_pca, f"t-SNE on PCA-Reduced {n_50}-D Data", t_pca),
]:
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y_sub, cmap="tab10", s=8, alpha=0.7)
    ax.set_title(f"{title}\n(fit time: {dt:.1f}s)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
fig.colorbar(sc, ax=axes, label="Digit", ticks=range(10), fraction=0.025)
plt.savefig(f"{OUT}/05_tsne_raw_vs_pca.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/05_tsne_raw_vs_pca.png")

perplexities = [5, 15, 30, 50, 100]
perplexities = [p for p in perplexities if p < TSNE_N]  # perplexity must be < n_samples
fig, axes = plt.subplots(1, len(perplexities), figsize=(4.5 * len(perplexities), 4.5))
if len(perplexities) == 1:
    axes = [axes]
perp_times = {}
for ax, p in zip(axes, perplexities):
    emb_p, dt = run_tsne(X_sub_pca50, p, "PCA-50")
    perp_times[p] = dt
    sc = ax.scatter(emb_p[:, 0], emb_p[:, 1], c=y_sub, cmap="tab10", s=6, alpha=0.7)
    ax.set_title(f"Perplexity = {p}\n({dt:.1f}s)")
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("t-SNE Embeddings Across Perplexity Values (input: PCA-50 features)", y=1.03)
plt.tight_layout()
plt.savefig(f"{OUT}/06_tsne_perplexity_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {OUT}/06_tsne_perplexity_comparison.png")

print("\n" + "=" * 70)
print("PART D: DIGIT RECOGNITION ANALYSIS (KNN & SVM)")
print("=" * 70)

CLF_N = min(8000, n_samples)
clf_idx = rng.choice(n_samples, CLF_N, replace=False)
X_clf_raw = X_norm[clf_idx]
X_clf_pca = X_pca_50d[clf_idx]
y_clf = y[clf_idx]

Xtr_raw, Xte_raw, Xtr_pca, Xte_pca, ytr, yte = train_test_split(
    X_clf_raw, X_clf_pca, y_clf, test_size=0.25, random_state=RNG, stratify=y_clf
)
print(f"\nClassifier subsample size: {CLF_N}  (train={len(ytr)}, test={len(yte)})")

results = []

def evaluate(name, model, Xtr, Xte, ytr, yte, feature_desc):
    t0 = time.time()
    model.fit(Xtr, ytr)
    train_time = time.time() - t0

    t0 = time.time()
    pred = model.predict(Xte)
    predict_time = time.time() - t0

    acc = accuracy_score(yte, pred)
    print(f"\n[{name} | {feature_desc}]")
    print(f"  Accuracy       : {acc*100:.2f}%")
    print(f"  Training time  : {train_time:.3f}s")
    print(f"  Prediction time: {predict_time:.3f}s ({predict_time/len(yte)*1000:.3f} ms/sample)")
    results.append({
        "Classifier": name, "Features": feature_desc, "Dimensions": Xtr.shape[1],
        "Accuracy": acc, "Train_Time_s": train_time, "Predict_Time_s": predict_time,
    })
    return pred

pred_knn_raw = evaluate("KNN (k=5)", KNeighborsClassifier(n_neighbors=5),
                         Xtr_raw, Xte_raw, ytr, yte, f"Raw {n_features}-D pixels")
pred_knn_pca = evaluate("KNN (k=5)", KNeighborsClassifier(n_neighbors=5),
                         Xtr_pca, Xte_pca, ytr, yte, f"PCA {n_50}-D")

pred_svm_raw = evaluate("SVM (RBF)", SVC(kernel="rbf", gamma="scale", random_state=RNG),
                         Xtr_raw, Xte_raw, ytr, yte, f"Raw {n_features}-D pixels")
pred_svm_pca = evaluate("SVM (RBF)", SVC(kernel="rbf", gamma="scale", random_state=RNG),
                         Xtr_pca, Xte_pca, ytr, yte, f"PCA {n_50}-D")

results_df = pd.DataFrame(results)
results_df.to_csv(f"{OUT}/classifier_results.csv", index=False)
print("\n" + "-" * 70)
print("SUMMARY TABLE")
print("-" * 70)
print(results_df.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics = [("Accuracy", "Accuracy", "{:.1%}"),
           ("Train_Time_s", "Training Time (s)", "{:.2f}s"),
           ("Predict_Time_s", "Prediction Time (s)", "{:.3f}s")]
labels = [f"{r.Classifier}\n{r.Features}" for r in results_df.itertuples()]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for ax, (col, title, fmt) in zip(axes, metrics):
    bars = ax.bar(labels, results_df[col], color=colors)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    for b, v in zip(bars, results_df[col]):
        ax.text(b.get_x() + b.get_width()/2, b.get_height(), fmt.format(v),
                 ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/07_classifier_comparison.png", dpi=150)
plt.close()
print(f"\nSaved: {OUT}/07_classifier_comparison.png")

plt.figure(figsize=(6, 5))
cm = confusion_matrix(yte, pred_svm_pca)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix: SVM on PCA-Reduced Features")
plt.tight_layout()
plt.savefig(f"{OUT}/08_confusion_matrix_svm_pca.png", dpi=150)
plt.close()
print(f"Saved: {OUT}/08_confusion_matrix_svm_pca.png")

print("\n" + "=" * 70)
print("ALL DONE. Figures + results CSV written to ./outputs/")
print("=" * 70)
