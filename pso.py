import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from sklearn.model_selection import train_test_split


class BinaryPSOMultiLabelFeatureSelection:

    def __init__(
        self,
        n_particles=20,
        n_iterations=30,
        alpha=0.99,
        w=0.7,
        c1=1.5,
        c2=1.5,
        random_state=42,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.random_state = random_state

        self.estimator = RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        )

    def _fitness(self, position):
        selected = np.flatnonzero(position)

        if len(selected) == 0:
            return 1.0

        model = clone(self.estimator)
        model.fit(self.X_train[:, selected], self.Y_train)
        predictions = model.predict(self.X_val[:, selected])
        score = f1_score(
            self.Y_val, predictions, average="micro", zero_division=0
        )
        error = 1.0 - score
        feature_ratio = len(selected) / self.n_features
        return self.alpha * error + (1.0 - self.alpha) * feature_ratio

    def fit(self, X, Y, feature_names=None):
        self.rng = np.random.default_rng(self.random_state)

        X = np.asarray(X)
        Y = np.asarray(Y)

        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError("X and Y must be 2D arrays.")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must have equal row counts.")
        if X.shape[1] == 0:
            raise ValueError("X must contain features.")
        if not np.isin(Y, [0, 1]).all():
            raise ValueError("Y must contain only 0 and 1.")

        self.n_features = X.shape[1]
        self.feature_names = (
            np.asarray(feature_names)
            if feature_names is not None
            else np.asarray([f"Feature{i + 1}" for i in range(self.n_features)])
        )

        if len(self.feature_names) != self.n_features:
            raise ValueError("Feature names must match X columns.")

        (
            self.X_train,
            self.X_val,
            self.Y_train,
            self.Y_val,
        ) = train_test_split(X, Y, test_size=0.25, random_state=self.random_state)

        n = self.n_particles
        d = self.n_features

        positions = self.rng.integers(0, 2, size=(n, d))
        velocities = self.rng.uniform(-1, 1, size=(n, d))

        for i in range(n):
            if positions[i].sum() == 0:
                positions[i, self.rng.integers(d)] = 1

        personal_best = positions.copy()
        personal_fitness = np.array([self._fitness(p) for p in positions])

        best_idx = np.argmin(personal_fitness)
        global_best = personal_best[best_idx].copy()
        global_fitness = personal_fitness[best_idx]

        for iteration in range(self.n_iterations):
            r1 = self.rng.random((n, d))
            r2 = self.rng.random((n, d))

            velocities = (
                self.w * velocities
                + self.c1 * r1 * (personal_best - positions)
                + self.c2 * r2 * (global_best - positions)
            )

            velocities = np.clip(velocities, -6, 6)

            probabilities = 1 / (1 + np.exp(-velocities))
            positions = (self.rng.random((n, d)) < probabilities).astype(int)

            for i in range(n):
                if positions[i].sum() == 0:
                    positions[i, self.rng.integers(d)] = 1

                score = self._fitness(positions[i])
                if score < personal_fitness[i]:
                    personal_best[i] = positions[i].copy()
                    personal_fitness[i] = score

                if score < global_fitness:
                    global_best = positions[i].copy()
                    global_fitness = score

            print(
                f"Iteration {iteration + 1}/{self.n_iterations} | "
                f"Fitness: {global_fitness:.5f} | "
                f"Features: {global_best.sum()}/{d}"
            )

        self.global_best_ = global_best
        self.selected_indices_ = np.flatnonzero(global_best)
        self.selected_features = self.feature_names[self.selected_indices_]
        self.best_fitness = global_fitness
        return self


# Generate sample multi-label dataset
np.random.seed(42)
n_samples = 200
n_features = 10

X = np.random.rand(n_samples, n_features)
Y = np.column_stack(
    [
        (X[:, 0] + X[:, 1] > 1).astype(int),
        (X[:, 2] + X[:, 3] > 1).astype(int),
        (X[:, 4] + X[:, 5] > 1).astype(int),
    ]
)

feature_names = [f"Feature{i + 1}" for i in range(n_features)]

# Run PSO feature selection
pso = BinaryPSOMultiLabelFeatureSelection(
    n_particles=10, n_iterations=10, alpha=0.99, random_state=42
)
pso.fit(X, Y, feature_names)

selected_indices = pso.selected_indices_

print("\n--- OPTIMIZATION RESULTS ---")
print("Best Feature Subset (Binary Mask):", pso.global_best_.tolist())
print("Best Fitness Score:", f"{pso.best_fitness:.5f}")

print("\n--- MAPPED BEST FEATURES ---")
for name, val in zip(pso.feature_names, pso.global_best_):
    status = "Selected (1)" if val == 1 else "Ignored (0)"
    print(f"{name}: {val} -> {status}")

print("\n--- SELECTED FEATURES ---")
print(pso.selected_features)
print("Feature indices (0-indexed):", selected_indices.tolist())
print(
    "Selected features (1-indexed):",
    (selected_indices + 1).tolist(),
)
print("Selected count:", len(selected_indices))
print("Original count:", n_features)

reduction = (1 - len(selected_indices) / n_features) * 100
print(f"Feature reduction: {reduction:.2f}%")

# Final model evaluation
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.25, random_state=42
)
X_train_selected = X_train[:, selected_indices]
X_val_selected = X_val[:, selected_indices]

final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X_train_selected, Y_train)
Y_pred = final_model.predict(X_val_selected)

print("\n--- FINAL MODEL EVALUATION ---")
print("Micro F1 Score:", f1_score(Y_val, Y_pred, average="micro", zero_division=0))
print("Macro F1 Score:", f1_score(Y_val, Y_pred, average="macro", zero_division=0))
print("Hamming Loss:", hamming_loss(Y_val, Y_pred))
print("Exact Match Accuracy:", accuracy_score(Y_val, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_val, Y_pred, zero_division=0))
#Preeth Rai
#1WN24CS205