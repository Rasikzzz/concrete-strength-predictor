# 🧱 Concrete Strength Predictor

A **deep learning MLP model** to predict the compressive strength of concrete based on its material composition.  
This project demonstrates regression using neural networks and can be extended for deployment or further experimentation in structural engineering applications.

---

## 📘 Overview

Concrete strength is a critical property in construction and engineering.  
This project uses a **feedforward neural network (MLP)** to predict the compressive strength of concrete given its ingredients and curing age.  

---

## 🧩 Features

- Predicts concrete compressive strength (MPa)  
- Implemented with **deep learning (MLP)**  
- Fully reproducible and ready for experimentation  
- Can be extended for deployment as a web app  

---

## 📊 Input Parameters

| Feature | Description |
|---------|-------------|
| Cement (kg/m³) | Amount of cement used |
| Blast Furnace Slag (kg/m³) | Optional additive |
| Fly Ash (kg/m³) | Optional additive |
| Water (kg/m³) | Mixing water |
| Superplasticizer (kg/m³) | Chemical additive |
| Coarse Aggregate (kg/m³) | Gravel/stones |
| Fine Aggregate (kg/m³) | Sand |
| Age (days) | Curing period |

**Target:** Compressive Strength (MPa)

---

## 🧠 Model Architecture

- **Type:** Multi-Layer Perceptron (MLP)  
- **Input Layer:** 8 features  
- **Hidden Layers:** 128 → 64 → 32 neurons (ReLU activation)  
- **Output Layer:** 1 neuron (linear activation)  

**Loss Function:** Mean Squared Error (MSE)  
**Optimizer:** Adam  

---

## ⚙️ Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
