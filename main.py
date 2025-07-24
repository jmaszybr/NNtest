from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import numpy as np

app = FastAPI()

# Pozwól na CORS (dla Twojej statycznej strony)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # Możesz wpisać tu konkretną domenę jak chcesz np. "https://twoja-strona.pl"
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

    clf = SVC(kernel=kernel, C=C, gamma=gamma)
    clf.fit(X, y)

    # Przygotuj siatkę do granicy
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = clf.predict(grid_points)

    return {
        "grid": {
            "X_grid": grid_points.tolist(),
            "predictions": predictions.tolist()
        },
        "accuracy": float(clf.score(X, y))
    }

@app.post("/train-nn")
def train_nn(req: TrainRequest):
    X = np.array(req.X)
    y = np.array(req.y)
    hidden_layers = tuple(req.params.get("hidden_layers", [20, 10]))
    activation = req.params.get("activation", "relu")
    max_iter = req.params.get("max_iter", 300)

    clf = MLPClassifier(hidden_layer_sizes=hidden_layers, activation=activation, max_iter=max_iter, random_state=42)
    clf.fit(X, y)

    # Przygotuj siatkę do granicy
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    predictions = clf.predict(grid_points)

    return {
        "grid": {
            "X_grid": grid_points.tolist(),
            "predictions": predictions.tolist()
        },
        "accuracy": float(clf.score(X, y))
    }
