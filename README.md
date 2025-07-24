# 🤖 ML Playground - Neural Networks vs SVM

Interactive web app for comparing Neural Networks and Support Vector Machines with real-time visualizations.

## ✨ Features

- **3 dataset types**: Moons, Circles, Clusters
- **Neural Networks**: Custom layers, activations, real-time loss curves
- **SVM**: Multiple kernels (RBF, Linear, Poly, Sigmoid)
- **Live charts**: Decision boundaries, training metrics, error visualization
- **Modern UI**: Responsive design with smooth animations

## 🚀 Quick Start

### Backend (FastAPI)
```bash
pip install fastapi uvicorn scikit-learn numpy
uvicorn main:app --reload
```

### Frontend
Open `index.html` in browser or:
```bash
python -m http.server 3000
```

## 📋 Requirements

```txt
fastapi==0.104.1
scikit-learn==1.3.2
numpy==1.24.4
uvicorn[standard]==0.24.0
```

## 🎮 Usage

1. **Generate data** - Choose dataset type and parameters
2. **Select model** - Neural Network or SVM
3. **Configure** - Set layers, activation, kernel, etc.
4. **Train & compare** - View real-time results and decision boundaries

## 🚀 Deploy

**Render.com**: Connect repo, set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

## 📊 What you'll see

- Training vs test accuracy
- Real-time loss curves  
- Decision boundary visualization
- Misclassified points highlighted
- Training time comparison

Built with FastAPI + Scikit-learn + Plotly.js
