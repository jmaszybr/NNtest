from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Możesz wpisać tu konkretną domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    dataset_type: str
    n_samples: int
    noise: Optional[float] = 0.1
    n_clusters: Optional[int] = 3
    cluster_std: Optional[float] = 1.0

class TrainRequest(BaseModel):
    X: List[List[float]]
    y: List[int]
    params: Dict[str, Any]

@app.get("/")
def root():
    return {"status": "Backend działa! :)"}

@app.post("/generate")
def generate(req: GenerateRequest):
    if req.dataset_type == "circles":
        X, y = make_circles(n_samples=req.n_samples, noise=req.noise, factor=0.5)
    elif req.dataset_type == "moons":
        X, y = make_moons(n_samples=req.n_samples, noise=req.noise)
    else:  # clusters
        X, y = make_blobs(n_samples=req.n_samples, centers=req.n_clusters, cluster_std=req.cluster_std)
    return {"X": X.tolist(), "y": y.tolist()}

@app.post("/train-svm")
def train_svm(req: TrainRequest):
    X = np.array(req.X)
    y = np.array(req.y)
    kernel = req.params.get("kernel", "rbf")
    C = req.params.get("C", 1.0)
    gamma = req.params.get("gamma", "scale")
    # Standaryzacja danych dla SVM
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X_std, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_std = scaler.transform(grid_points)
    predictions = clf.predict(grid_points_std)

    return {
        "grid": {
            "X_grid": grid_points.tolist(),
            "predictions": predictions.tolist()
        },
        "accuracy": float(clf.score(X_std, y)),
        "loss_curve": []
    }

@app.post("/train-nn")
def train_nn(req: TrainRequest):
    X = np.array(req.X)
    y = np.array(req.y)
    hidden_layers = tuple(req.params.get("hidden_layers", [20, 10]))
    activation = req.params.get("activation", "relu")
    max_iter = req.params.get("max_iter", 300)
    test_size = req.params.get("test_size", 0.2)  # Możesz przekazywać z frontu, domyślnie 0.2

    # Podział na train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    # Standaryzacja
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    # Trening NN z early stopping
    clf = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        max_iter=max_iter,
        random_state=42,
        solver="adam",
        verbose=False,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=15
    )
    clf.fit(X_train_std, y_train)
    loss_curve = getattr(clf, "loss_curve_", None)
    val_scores = getattr(clf, "validation_scores_", None)  # sklearn >=1.2

    # Predykcja na siatce do granicy decyzyjnej
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_std = scaler.transform(grid_points)
    predictions = clf.predict(grid_points_std)

    # Wyniki testowe
    y_test_pred = clf.predict(X_test_std)
    errors = np.where(y_test_pred != y_test)[0]

    return {
        "grid": {
            "X_grid": grid_points.tolist(),
            "predictions": predictions.tolist()
        },
        "accuracy_train": float(clf.score(X_train_std, y_train)),
        "accuracy_test": float(clf.score(X_test_std, y_test)),
        "loss_curve": list(loss_curve) if loss_curve is not None else [],
        "val_loss_curve": list(1-np.array(val_scores)) if val_scores is not None else [],
        "test_points": X_test.tolist(),
        "test_true": y_test.tolist(),
        "test_pred": y_test_pred.tolist(),
        "test_errors_idx": errors.tolist()
    }
