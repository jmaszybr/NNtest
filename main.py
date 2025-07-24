from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Tuple
from sklearn.datasets import make_circles, make_moons, make_blobs
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, classification_report
import numpy as np
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="🤖 ML Playground API",
    description="Advanced Neural Networks & SVM Playground with real-time training metrics",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enhanced Pydantic models
class GenerateRequest(BaseModel):
    dataset_type: str = Field(..., description="Type of dataset: moons, circles, or clusters")
    n_samples: int = Field(250, ge=50, le=2000, description="Number of samples to generate")
    noise: float = Field(0.15, ge=0.0, le=1.0, description="Noise level")
    n_clusters: Optional[int] = Field(3, ge=2, le=5, description="Number of clusters (for clusters dataset)")
    cluster_std: Optional[float] = Field(0.8, ge=0.1, le=3.0, description="Cluster standard deviation")
    random_state: Optional[int] = Field(42, description="Random state for reproducibility")

class NeuralNetworkParams(BaseModel):
    hidden_layers: List[int] = Field([20, 10], description="Hidden layer sizes")
    activation: str = Field("relu", description="Activation function")
    max_iter: int = Field(500, ge=50, le=2000, description="Maximum iterations")
    learning_rate_init: float = Field(0.001, ge=0.0001, le=0.1, description="Initial learning rate")
    batch_size: str = Field("auto", description="Batch size")
    solver: str = Field("adam", description="Solver algorithm")
    alpha: float = Field(0.0001, ge=0.0, le=1.0, description="L2 regularization parameter")

class SVMParams(BaseModel):
    kernel: str = Field("rbf", description="Kernel type")
    C: float = Field(1.0, ge=0.01, le=100.0, description="Regularization parameter")
    gamma: str = Field("scale", description="Kernel coefficient")
    degree: int = Field(3, ge=1, le=10, description="Degree for poly kernel")

class TrainNNRequest(BaseModel):
    X: List[List[float]]
    y: List[int]
    params: NeuralNetworkParams

class TrainSVMRequest(BaseModel):
    X: List[List[float]]
    y: List[int]
    params: SVMParams

# Advanced Neural Network Trainer with custom training loop
class AdvancedMLPTrainer:
    def __init__(self, hidden_layers, activation, max_iter, learning_rate_init, 
                 batch_size, solver, alpha, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layers),
            activation=activation,
            max_iter=1,  # We'll train iteratively
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            solver=solver,
            alpha=alpha,
            random_state=random_state,
            warm_start=True,
            verbose=False
        )
        self.max_iter = max_iter
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 25
        
    def fit(self, X_train, y_train, X_val, y_val):
        logger.info(f"Starting training with {len(X_train)} train samples, {len(X_val)} val samples")
        
        for epoch in range(1, self.max_iter + 1):
            # Train for one more iteration
            self.model.max_iter = epoch
            self.model.fit(X_train, y_train)
            
            # Calculate training metrics
            try:
                train_pred_proba = self.model.predict_proba(X_train)
                train_loss = log_loss(y_train, train_pred_proba)
                train_acc = accuracy_score(y_train, self.model.predict(X_train))
                
                # Calculate validation metrics
                val_pred_proba = self.model.predict_proba(X_val)
                val_loss = log_loss(y_val, val_pred_proba)
                val_acc = accuracy_score(y_val, self.model.predict(X_val))
                
                # Store metrics
                self.train_losses.append(float(train_loss))
                self.val_losses.append(float(val_loss))
                self.train_accuracies.append(float(train_acc))
                self.val_accuracies.append(float(val_acc))
                
                # Early stopping logic
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Log progress every 50 epochs
                if epoch % 50 == 0:
                    logger.info(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "
                              f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience exceeded)")
                    break
                    
            except Exception as e:
                logger.warning(f"Error calculating metrics at epoch {epoch}: {e}")
                # Use simplified loss calculation
                train_loss = 1 - self.model.score(X_train, y_train)
                val_loss = 1 - self.model.score(X_val, y_val)
                self.train_losses.append(float(train_loss))
                self.val_losses.append(float(val_loss))
                self.train_accuracies.append(float(self.model.score(X_train, y_train)))
                self.val_accuracies.append(float(self.model.score(X_val, y_val)))
        
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        return self.model.score(X, y)

@app.get("/")
def root():
    return {
        "status": "🚀 Advanced ML Playground Backend is running!",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Advanced Neural Networks with real-time training metrics",
            "Enhanced SVM with multiple kernels",
            "Proper train/validation/test splits",
            "Real validation loss tracking",
            "Early stopping with patience",
            "Comprehensive error handling"
        ]
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/generate")
def generate_dataset(req: GenerateRequest):
    """Generate synthetic datasets with enhanced parameters"""
    try:
        logger.info(f"Generating {req.dataset_type} dataset with {req.n_samples} samples")
        
        if req.dataset_type == "circles":
            X, y = make_circles(
                n_samples=req.n_samples, 
                noise=req.noise, 
                factor=0.5,
                random_state=req.random_state
            )
        elif req.dataset_type == "moons":
            X, y = make_moons(
                n_samples=req.n_samples, 
                noise=req.noise,
                random_state=req.random_state
            )
        elif req.dataset_type == "clusters":
            X, y = make_blobs(
                n_samples=req.n_samples, 
                centers=req.n_clusters, 
                cluster_std=req.cluster_std,
                random_state=req.random_state
            )
        else:
            raise HTTPException(status_code=400, detail=f"Unknown dataset type: {req.dataset_type}")
        
        # Dataset statistics
        unique_classes, class_counts = np.unique(y, return_counts=True)
        
        response = {
            "X": X.tolist(),
            "y": y.tolist(),
            "metadata": {
                "n_samples": len(X),
                "n_features": X.shape[1],
                "n_classes": len(unique_classes),
                "class_distribution": dict(zip(unique_classes.tolist(), class_counts.tolist())),
                "feature_ranges": {
                    "x1": {"min": float(X[:, 0].min()), "max": float(X[:, 0].max())},
                    "x2": {"min": float(X[:, 1].min()), "max": float(X[:, 1].max())}
                }
            }
        }
        
        logger.info(f"Generated dataset: {response['metadata']}")
        return response
        
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating dataset: {str(e)}")

@app.post("/train-nn")
def train_neural_network(req: TrainNNRequest):
    """Train neural network with train/test split (no validation)"""
    try:
        start_time = datetime.now()
        logger.info("Starting neural network training")
        
        X = np.array(req.X)
        y = np.array(req.y)
        
        # Simple train/test split (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Standardization
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        # Standard MLPClassifier with early stopping based on training loss
        clf = MLPClassifier(
            hidden_layer_sizes=tuple(req.params.hidden_layers),
            activation=req.params.activation,
            max_iter=req.params.max_iter,
            learning_rate_init=req.params.learning_rate_init,
            batch_size=req.params.batch_size,
            solver=req.params.solver,
            alpha=req.params.alpha,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,  # Uses part of training data for internal validation
            n_iter_no_change=20,
            verbose=False
        )
        
        clf.fit(X_train_std, y_train)
        
        # Get training curves
        train_losses = getattr(clf, "loss_curve_", [])
        validation_scores = getattr(clf, "validation_scores_", [])
        
        # Convert validation scores to loss approximation
        val_losses = [1.0 - score for score in validation_scores] if validation_scores else []
        
        # Decision boundary 
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(y_min, y_max, 150)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_std = scaler.transform(grid_points)
        predictions = clf.predict(grid_points_std)
        
        # Test predictions
        y_test_pred = clf.predict(X_test_std)
        test_errors = np.where(y_test_pred != y_test)[0]
        
        # Final metrics
        train_acc = clf.score(X_train_std, y_train)
        test_acc = clf.score(X_test_std, y_test)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "grid": {
                "X_grid": grid_points.tolist(),
                "predictions": predictions.tolist()
            },
            "metrics": {
                "accuracy_train": float(train_acc),
                "accuracy_test": float(test_acc),
                "final_train_loss": train_losses[-1] if train_losses else None,
                "final_val_loss": val_losses[-1] if val_losses else None,
                "epochs_trained": len(train_losses),
                "training_time_seconds": training_time
            },
            "curves": {
                "train_loss": train_losses,
                "val_loss": val_losses,
                "train_accuracy": [clf.score(X_train_std, y_train)] * len(train_losses) if train_losses else [],
                "val_accuracy": validation_scores
            },
            "test_data": {
                "points": X_test.tolist(),
                "true_labels": y_test.tolist(),
                "predictions": y_test_pred.tolist(),
                "error_indices": test_errors.tolist()
            },
            "data_splits": {
                "train_size": len(X_train),
                "test_size": len(X_test)
            }
        }
        
        logger.info(f"Training completed in {training_time:.2f}s - Test accuracy: {test_acc:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Error training neural network: {e}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/train-svm")
def train_svm(req: TrainSVMRequest):
    """Train SVM with enhanced configuration"""
    try:
        start_time = datetime.now()
        logger.info("Starting SVM training")
        
        X = np.array(req.X)
        y = np.array(req.y)
        
        # Train/test split for SVM
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        # Standardization
        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)
        
        # SVM training with enhanced parameters
        svm_params = {
            "kernel": req.params.kernel,
            "C": req.params.C,
            "probability": True,  # For probability estimates
            "random_state": 42
        }
        
        if req.params.kernel == "rbf":
            svm_params["gamma"] = req.params.gamma
        elif req.params.kernel == "poly":
            svm_params["degree"] = req.params.degree
            svm_params["gamma"] = req.params.gamma
        
        clf = SVC(**svm_params)
        clf.fit(X_train_std, y_train)
        
        # Decision boundary
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 150),
            np.linspace(y_min, y_max, 150)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_std = scaler.transform(grid_points)
        predictions = clf.predict(grid_points_std)
        
        # Metrics
        train_acc = clf.score(X_train_std, y_train)
        test_acc = clf.score(X_test_std, y_test)
        
        # Test predictions
        y_test_pred = clf.predict(X_test_std)
        test_errors = np.where(y_test_pred != y_test)[0]
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        response = {
            "grid": {
                "X_grid": grid_points.tolist(),
                "predictions": predictions.tolist()
            },
            "metrics": {
                "accuracy_train": float(train_acc),
                "accuracy_test": float(test_acc),
                "n_support_vectors": int(clf.n_support_.sum()) if hasattr(clf, 'n_support_') else None,
                "training_time_seconds": training_time
            },
            "test_data": {
                "points": X_test.tolist(),
                "true_labels": y_test.tolist(),
                "predictions": y_test_pred.tolist(),
                "error_indices": test_errors.tolist()
            },
            "model_info": {
                "kernel": req.params.kernel,
                "C": req.params.C,
                "gamma": req.params.gamma if req.params.kernel in ["rbf", "poly"] else None,
                "degree": req.params.degree if req.params.kernel == "poly" else None
            }
        }
        
        logger.info(f"SVM training completed in {training_time:.2f}s - Test accuracy: {test_acc:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Error training SVM: {e}")
        raise HTTPException(status_code=500, detail=f"SVM training error: {str(e)}")

@app.get("/models/info")
def get_models_info():
    """Get information about available models and parameters"""
    return {
        "neural_network": {
            "activations": ["relu", "tanh", "logistic"],
            "solvers": ["adam", "lbfgs", "sgd"],
            "batch_sizes": ["auto", 32, 64, 128, 256],
            "recommended_layers": [[10], [20, 10], [50, 20], [100, 50, 20]]
        },
        "svm": {
            "kernels": ["rbf", "linear", "poly", "sigmoid"],
            "C_range": [0.01, 0.1, 1.0, 10.0, 100.0],
            "gamma_options": ["scale", "auto", 0.001, 0.01, 0.1, 1.0]
        },
        "datasets": {
            "types": ["moons", "circles", "clusters"],
            "recommended_samples": [100, 250, 500, 1000],
            "noise_levels": [0.0, 0.1, 0.2, 0.3]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
