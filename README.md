```markdown
# Lab 3: Neural Networks - Feedforward Networks and the Training Process

**Student:** Rafiatou Malam Ali

**Student ID:** 30782027

**Course:** Introduction to Artificial Intelligence

**Date:** 28/07/2026

---

## Overview

This lab bridges the gap between classical ML and deep learning. Instead of calling `.fit()` and trusting scikit-learn, I opened the black box: building a neural network from a single neuron upward, watching gradient descent learn, and then training a full feedforward network (MLP) in PyTorch on the obesity dataset from Lab 2.

By the end, I can explain and demonstrate with my own plots what each of these really does: **weights, activation functions, loss, gradients, learning rate, epochs, batches, and overfitting**.

---

## Repository Structure

```
lab-3/
├── lab_3.ipynb                 # Main notebook with all implementations
├── AI_USE_DECLARATION.md       # AI use declaration form
├── README.md                   # This file              
```

---

## Part 1: Inside a Neural Network - From One Neuron to Gradient Descent

### 1.1 - A Single Neuron is Just a Line

**What I did:**
- Generated 100 noisy samples of `y = sin(x)` on `[-3, 3]`
- Implemented a single neuron `y_hat = w * x + b`
- Trained it with manual gradient descent for 200 steps
- Plotted the loss curve and the fitted line

**Key learning:** A single neuron can only fit a straight line. The loss stops improving but the fit is clearly bad. This is **underfitting** because the model is too simple to capture the sine wave's curvature.

### 1.2 - Hidden Layers and Activations: Why Depth Helps

**What I did:**
- Built a tiny network by hand: 1 input → 8 neurons (tanh) → 1 output
- Implemented forward pass, backward pass (chain rule), and parameter updates
- Trained for 3000 steps
- Plotted the loss curve and the final fit
- Re-ran with tanh removed to see what happens

**Key learning:** 
- With tanh, the network can fit the sine wave
- Without tanh, it collapses to a linear model (stack of linear layers = one linear layer)
- Hidden neurons each learn different features (sigmoid-like activations at different offsets)
- **Backpropagation** = computing gradients from the output backward through the network using the chain rule

### 1.3 - The Learning Rate: Too Cold, Too Hot, Just Right

**What I did:**
- Wrapped training in a function
- Ran with three learning rates: 0.001, 0.05, 1.0
- Plotted loss curves on the same axes

**Key learning:**
- **Too cold (0.001):** Loss decreases very slowly
- **Just right (0.05):** Loss decreases steadily and reaches a good minimum
- **Too hot (1.0):** Loss explodes or oscillates because steps overshoot the minimum

---

## Part 2: A Real Feedforward Network in PyTorch

### 2.1 - Preprocess and Split

**What I did:**
- Loaded the obesity dataset (NObeyesdad) - 2,111 samples, 17 features
- Encoded categorical variables (binary → 0/1, multi-class → ordinal/one-hot)
- Encoded target to integers 0..6 (7 classes)
- Stratified train/validation/test split (60/20/20)
- StandardScaler on training set only
- Converted to tensors and DataLoaders (batch_size=32)

**Key learning:**
- `CrossEntropyLoss` wants integer class labels, not one-hot
- Neural networks care about feature scaling because gradients depend on input values
- `shuffle=True` prevents the model from seeing data in the same order each epoch

### 2.2 - Design the Network

**What I did:**
- Defined `FeedForwardNet` class with input → hidden layers (ReLU) → 7 output logits
- Used hidden sizes: (64, 32)
- Counted trainable parameters: 3,655 total

**Parameter Count Verification:**

| Layer | Weights | Biases | Total |
|-------|---------|--------|-------|
| Input → Hidden (20→64) | 1,280 | 64 | 1,344 |
| Hidden → Hidden (64→32) | 2,048 | 32 | 2,080 |
| Hidden → Output (32→7) | 224 | 7 | 231 |
| **Total** | **3,552** | **103** | **3,655** |

### 2.3 - The Training Loop

**What I did:**
- Defined loss: `nn.CrossEntropyLoss()`
- Defined optimizer: `torch.optim.Adam(model.parameters(), lr=1e-3)`
- Trained for 50 epochs
- Saved best model (highest validation accuracy)
- Plotted loss and accuracy curves

**Final Results:**
- Train Accuracy: 100.00%
- Validation Accuracy: 93.13%
- Train-Val Gap: 6.87%

### 2.4 - Experiments

**Experiment A - Capacity:**
- Tiny (8,): 80.33% validation accuracy → underfitting
- Default (64, 32): 94.08% validation accuracy → well-fitted
- Large (256, 128, 64): 94.55% validation accuracy → diminishing returns

**Experiment B - Learning Rate:**
- lr = 1e-4: 74.41% accuracy (too slow)
- lr = 1e-3: 93.60% accuracy (just right)
- lr = 1e-1: 45.97% accuracy (exploded)

**Experiment C - Regularization (Dropout):**
- Without Dropout: Train 100%, Val 94.08%, Gap 5.92%
- With Dropout (0.3): Train 97.08%, Val 92.89%, Gap 4.19%
- Dropout reduced the train-val gap from 5.92% to 4.19%

### 2.5 - Final Evaluation and Showdown with Lab 2

**Neural Network Results:**
- Test Accuracy: 93.38%
- Test Macro-F1: 93.26%

**Comparison Table:**

| Model | Test Accuracy | Macro-F1 | Training Time | Parameters |
|-------|--------------|----------|---------------|------------|
| Logistic Regression | 89.36% | 88.95% | 0.5s | ~20 |
| Random Forest (Lab 2) | 98.82% | 98.77% | 4.1s | 100 trees |
| Neural Network (MLP) | 93.38% | 93.26% | 50s | 3,655 |

**Winner:**  Random Forest wins on both Accuracy and F1!

**Conclusion:** For this small tabular dataset (2,111 samples), Random Forest was faster, more accurate, and more interpretable. The Neural Network was overkill and not worth the extra effort.

---

## Section 3: Reflection

### 1. The black box, opened:

Three most important things happening inside `loss.backward()` and `optimizer.step()`:

1. **Forward Pass:** Data flows through the network layer by layer to make predictions
2. **Backward Pass (Backpropagation):** Calculates gradients of the loss with respect to every weight using the chain rule
3. **Weight Update:** Uses gradients to update weights in the direction that reduces loss (`w = w - lr * dw`)

### 2. Hyperparameters:

I would tune the **learning rate** first. It's the step size of gradient descent:
- Too small = slow learning (wasted time)
- Too large = unstable training (can explode)

### 3. Overfitting:

The biggest train-val gap was with the Large network without dropout (5.92%). Dropout reduced it to 4.19% by randomly turning off neurons during training, forcing the network to learn more robust representations.

### 4. Looking ahead:

A plain MLP struggles with images because:
- Too many parameters (10,000 pixels × 64 neurons = 640,000 weights!)
- Loses spatial information (flattens 2D images)
- No translation invariance (learns patterns for each position separately)

Better architectures (CNNs) should have:
- **Local connectivity:** Neurons only look at small regions
- **Weight sharing:** Same filters slide across the image
- **Hierarchical learning:** Simple patterns → complex patterns

---

## Requirements

- Python 3.7+
- Jupyter Notebook or Google Colab
- Required libraries: numpy, pandas, matplotlib, torch, sklearn

---

## GitHub Repository

[https://github.com/rafiatoumalamali-cell/Rafiatou_Malam_ali_AI_lab-3.git](https://github.com/rafiatoumalamali-cell/Rafiatou_Malam_ali_AI_lab-3.git)

---

## Declaration

I declare that this work is my own. AI tools were used as declared in `AI_USE_DECLARATION.md` and do not exceed 25% of the total work.

---

**Signature:** RMA

**Date:** 28/07/2026
```
