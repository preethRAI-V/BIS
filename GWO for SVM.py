import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# CONFIG  (set these BEFORE running; do not
# change them after seeing results)
# ==========================================

TEST_SIZE = 0.3          # 30% test = ~90 patients (less noisy than 60)
SPLIT_SEED = 123         # arbitrary seed for the main split
RUN_MULTI_SPLIT = True   # repeated-split evaluation (mean +/- std)
N_SPLITS_EVAL = 10       # number of random splits for that evaluation

# ==========================================
# 1. LOAD DATASET
# ==========================================

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

columns = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal", "target"
]

df = pd.read_csv(url, names=columns)

df.replace("?", np.nan, inplace=True)
df.dropna(inplace=True)
df = df.astype(float)

# Convert into binary classification
df["target"] = (df["target"] > 0).astype(int)

X = df.drop("target", axis=1)
y = df["target"]

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(y.value_counts())


# ==========================================
# 2. GREY WOLF OPTIMIZER
# ==========================================

class GWO:

    def __init__(self, fitness, dim, lb, ub,
                 wolves=20, iterations=40,
                 init_positions=None, seed=42, verbose=True):

        self.fitness = fitness          # returns (objective, cv_accuracy)
        self.dim = dim
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)
        self.wolves = wolves
        self.iterations = iterations
        self.init_positions = init_positions
        self.seed = seed
        self.verbose = verbose

    def optimize(self):

        rng = np.random.default_rng(self.seed)

        positions = rng.uniform(
            self.lb, self.ub,
            (self.wolves, self.dim)
        )

        # Seed the default SVM as a starting wolf
        if self.init_positions is not None:
            for k, p in enumerate(self.init_positions):
                positions[k] = np.clip(p, self.lb, self.ub)

        best_obj = np.inf
        best_pos = None
        best_acc = None
        convergence = []

        # iterations + 1 so the final update is evaluated too
        for t in range(self.iterations + 1):

            results = [self.fitness(p) for p in positions]
            objs = np.array([r[0] for r in results])
            accs = np.array([r[1] for r in results])

            order = np.argsort(objs, kind="stable")

            if objs[order[0]] < best_obj:
                best_obj = objs[order[0]]
                best_pos = positions[order[0]].copy()
                best_acc = accs[order[0]]

            convergence.append(best_acc)

            if self.verbose:
                print(
                    f"Iteration {t}/{self.iterations} "
                    f"| Best CV Accuracy: {best_acc:.4f}"
                )

            if t == self.iterations:
                break

            alpha, beta, delta = positions[order[:3]]

            a = 2 - 2 * t / self.iterations

            new_positions = np.empty_like(positions)

            for i in range(self.wolves):

                candidates = []

                for leader in (alpha, beta, delta):

                    r1 = rng.random(self.dim)
                    r2 = rng.random(self.dim)

                    A = 2 * a * r1 - a
                    C = 2 * r2

                    D = np.abs(C * leader - positions[i])

                    candidates.append(leader - A * D)

                new_positions[i] = np.mean(candidates, axis=0)

            positions = np.clip(new_positions, self.lb, self.ub)

            # Keep the global best wolf in the population
            positions[0] = best_pos.copy()

        return best_pos, best_acc, convergence


# ==========================================
# 3. TUNE SVM WITH GWO (reusable)
# ==========================================

def tune_svm_with_gwo(X_tr, y_tr, wolves=20, iterations=40,
                      cv_repeats=3, seed=42, verbose=True):
    """Tune (C, gamma) of an RBF SVM using ONLY the given training data."""

    X_np = np.asarray(X_tr, dtype=float)
    y_np = np.asarray(y_tr)

    cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=cv_repeats, random_state=seed
    )

    # Scaling is fitted on each training fold only (no leakage) and does
    # not depend on (C, gamma), so it is precomputed once.
    folds = []
    for tr_idx, va_idx in cv.split(X_np, y_np):
        sc = StandardScaler().fit(X_np[tr_idx])
        folds.append((
            sc.transform(X_np[tr_idx]), y_np[tr_idx],
            sc.transform(X_np[va_idx]), y_np[va_idx]
        ))

    def cv_accuracy(C, gamma):
        accs = []
        for Xtr, ytr, Xva, yva in folds:
            model = SVC(C=C, gamma=gamma, kernel="rbf")
            model.fit(Xtr, ytr)
            accs.append((model.predict(Xva) == yva).mean())
        return float(np.mean(accs))

    # gamma="scale" = 1 / (n_features * X.var()) on scaled data
    gamma_scale = 1.0 / (
        X_np.shape[1] * StandardScaler().fit_transform(X_np).var()
    )
    default_pos = np.array([np.log10(1.0), np.log10(gamma_scale)])

    def fitness(params):
        C = 10 ** params[0]
        gamma = 10 ** params[1]
        acc = cv_accuracy(C, gamma)

        # Tiny tie-breaker toward the default region
        # (smaller than one CV accuracy step)
        penalty = 1e-4 * (
            abs(params[0] - default_pos[0]) +
            abs(params[1] - default_pos[1])
        )
        return (1 - acc) + penalty, acc

    gwo = GWO(
        fitness=fitness,
        dim=2,
        lb=[-2, -4],        # log10(C) in [0.01, 1000]
        ub=[3, 1],          # log10(gamma) in [1e-4, 10]
        wolves=wolves,
        iterations=iterations,
        init_positions=[default_pos],
        seed=seed,
        verbose=verbose
    )

    best_pos, best_acc, convergence = gwo.optimize()

    return {
        "C": 10 ** best_pos[0],
        "gamma": 10 ** best_pos[1],
        "cv_acc": best_acc,
        "default_cv_acc": cv_accuracy(1.0, "scale"),
        "gamma_scale": gamma_scale,
        "convergence": convergence
    }


# ==========================================
# 4. MODEL DEFINITIONS
# ==========================================

def build_models(best_C, best_gamma):
    return {

        "SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf")
        ),

        "GWO-SVM": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=best_C, gamma=best_gamma)
        ),

        "Decision Tree": DecisionTreeClassifier(random_state=42),

        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),

        "KNN": make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(n_neighbors=5)
        ),

        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000)
        ),

        "Naive Bayes": GaussianNB()
    }


# ==========================================
# 5. MAIN SPLIT + GWO TUNING
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SPLIT_SEED,
    stratify=y
)

print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")
print("\nStarting Grey Wolf Optimization...")
start = time.time()

tuned = tune_svm_with_gwo(
    X_train, y_train,
    wolves=20, iterations=40, cv_repeats=3, seed=42, verbose=True
)

print(f"\nGWO finished in {time.time() - start:.1f} s")

best_C = tuned["C"]
best_gamma = tuned["gamma"]
best_cv_acc = tuned["cv_acc"]
default_cv_acc = tuned["default_cv_acc"]
convergence = tuned["convergence"]

print("\n===== SVM HYPERPARAMETER COMPARISON =====")

print("Default SVM:")
print("C = 1.0")
print("Gamma = scale (~", round(tuned["gamma_scale"], 5), ")")
print(f"CV accuracy = {default_cv_acc * 100:.2f}%")

print("\nGWO Optimized SVM:")
print("C =", best_C)
print("Gamma =", best_gamma)
print(f"CV accuracy = {best_cv_acc * 100:.2f}%")


# ==========================================
# 6. TRAIN AND EVALUATE ON THE MAIN SPLIT
# ==========================================

models = build_models(best_C, best_gamma)

results = {}
predictions = {}

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy
    predictions[name] = y_pred

    print("\n----------------------------")
    print(name)
    print("----------------------------")

    print("Test Accuracy:", round(accuracy * 100, 2), "%")

    print(classification_report(
        y_test, y_pred, zero_division=0
    ))


# ==========================================
# 7. ACCURACY COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(11, 6))

names = list(results.keys())
accuracies = [results[name] * 100 for name in names]

colors = [
    "steelblue" if name != "GWO-SVM" else "green"
    for name in names
]

bars = plt.bar(names, accuracies, color=colors)

for bar, value in zip(bars, accuracies):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{value:.2f}%",
        ha="center"
    )

plt.title("Classification Accuracy Comparison (main split)")
plt.xlabel("Machine Learning Algorithms")
plt.ylabel("Test Accuracy (%)")
plt.ylim(0, 110)
plt.xticks(rotation=25, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()


# ==========================================
# 8. SVM VS GWO-SVM (CV AND TEST)
# ==========================================

plt.figure(figsize=(8, 5))

labels = ["Standard SVM", "GWO-SVM"]
cv_vals = [default_cv_acc * 100, best_cv_acc * 100]
test_vals = [results["SVM"] * 100, results["GWO-SVM"] * 100]

x = np.arange(len(labels))
w = 0.35

b1 = plt.bar(x - w / 2, cv_vals, w, label="CV Accuracy (train)",
             color="steelblue")
b2 = plt.bar(x + w / 2, test_vals, w, label="Test Accuracy",
             color="green")

for bars_ in (b1, b2):
    for bar in bars_:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{bar.get_height():.2f}%",
            ha="center"
        )

plt.xticks(x, labels)
plt.title("Standard SVM vs GWO Optimized SVM")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 110)
plt.legend()
plt.grid(axis="y", alpha=0.3)
plt.show()


# ==========================================
# 9. GWO CONVERGENCE GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(0, len(convergence)),
    np.array(convergence) * 100,
    marker="o",
    color="green",
    label="Best GWO CV Accuracy"
)

plt.axhline(
    default_cv_acc * 100,
    color="blue",
    linestyle="--",
    label="Standard SVM CV Accuracy"
)

plt.title("SVM Optimization Using Grey Wolf Optimization")
plt.xlabel("GWO Iterations")
plt.ylabel("CV Accuracy (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# ==========================================
# 10. GWO-SVM CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, predictions["GWO-SVM"])

plt.figure(figsize=(6, 5))

plt.imshow(cm, interpolation="nearest", cmap="Greens")

plt.title("GWO-SVM Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks([0, 1], ["No Disease", "Disease"])
plt.yticks([0, 1], ["No Disease", "Disease"])

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center", va="center",
            color="black", fontsize=14
        )

plt.colorbar()
plt.tight_layout()
plt.show()


# ==========================================
# 11. FINAL RESULTS TABLE (main split)
# ==========================================

final_results = pd.DataFrame({
    "Algorithm": results.keys(),
    "Accuracy (%)": [round(v * 100, 2) for v in results.values()]
})

final_results = final_results.sort_values(
    by="Accuracy (%)",
    ascending=False
)

print("\nFINAL MODEL COMPARISON (main split)")
print(final_results.to_string(index=False))


# ==========================================
# 12. REPEATED-SPLIT EVALUATION (mean +/- std)
# ==========================================
# For each of N random splits: run GWO on that split's TRAIN part only,
# then score every model on that split's TEST part. This removes the
# luck of a single split and gives a trustworthy comparison.
# (Uses a lighter GWO: 12 wolves, 10 iterations, 2x5-fold CV.)

if RUN_MULTI_SPLIT:

    print(f"\n\n===== REPEATED-SPLIT EVALUATION "
          f"({N_SPLITS_EVAL} splits) =====")

    rows = []

    for s in range(N_SPLITS_EVAL):

        Xtr, Xte, ytr, yte = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            stratify=y,
            random_state=1000 + s
        )

        t_ = tune_svm_with_gwo(
            Xtr, ytr,
            wolves=12, iterations=10, cv_repeats=2,
            seed=s, verbose=False
        )

        accs = {}
        for name, model in build_models(t_["C"], t_["gamma"]).items():
            model.fit(Xtr, ytr)
            accs[name] = accuracy_score(yte, model.predict(Xte)) * 100

        rows.append(accs)

        print(
            f"Split {s + 1}/{N_SPLITS_EVAL} | "
            f"SVM {accs['SVM']:.2f}% | GWO-SVM {accs['GWO-SVM']:.2f}%"
        )

    multi = pd.DataFrame(rows)

    summary = pd.DataFrame({
        "Mean Acc (%)": multi.mean().round(2),
        "Std (%)": multi.std().round(2)
    }).sort_values("Mean Acc (%)", ascending=False)

    print("\nMEAN +/- STD TEST ACCURACY OVER SPLITS")
    print(summary.to_string())

    diff = multi["GWO-SVM"] - multi["SVM"]
    print(
        f"\nGWO-SVM vs default SVM over {N_SPLITS_EVAL} splits: "
        f"wins {(diff > 0).sum()}, ties {(diff == 0).sum()}, "
        f"losses {(diff < 0).sum()}, "
        f"mean difference {diff.mean():+.2f} points"
    )

    plt.figure(figsize=(11, 6))

    order_names = list(summary.index)
    means = summary["Mean Acc (%)"].values
    stds = summary["Std (%)"].values

    plt.bar(
        order_names, means, yerr=stds, capsize=5,
        color=["green" if n == "GWO-SVM" else "steelblue"
               for n in order_names]
    )

    for i, (m, sd) in enumerate(zip(means, stds)):
        plt.text(i, m + sd + 0.8, f"{m:.2f}", ha="center")

    plt.title(f"Mean Test Accuracy over {N_SPLITS_EVAL} Random Splits "
              f"(error bars = std)")
    plt.ylabel("Test Accuracy (%)")
    plt.ylim(0, 110)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    #Preeth Rai
    #1WN24CS205