<div align="center">
  <h1>🚀 Nexus ML</h1>
  <p><strong>A Pure Rust Machine Learning Framework bridging the Research-to-Production gap.</strong></p>
</div>

---

## 📖 What is it?

**Nexus ML** is a fully-featured, from-scratch Machine Learning framework written entirely in Rust. 

The current ML ecosystem suffers from a massive **"Research-to-Production Gap."** Data scientists design and train models in Python (PyTorch/TensorFlow) because of its excellent ergonomics. However, deploying these models at scale in production often requires rewriting inference code in C++, exporting to fragile ONNX graphs, or dealing with the operational nightmare of scaling Python environments.

**Nexus ML changes this.** By leveraging Rust's performance, memory safety, and powerful procedural macros, it provides the expressiveness and ease-of-use of PyTorch alongside the raw speed and zero-cost abstraction of a compiled binary. 

*Write your models like Python. Run them like C++.*

---

## 🎯 For Whom?

- **ML Engineers & MLOps:** Professionals tired of the friction between training environments (Python) and deployment environments (C++/CUDA).
- **Rust Developers:** Systems engineers who want to build and integrate AI directly into their Rust applications without relying on FFI bindings or Python wrappers.
- **Startups & Edge Computing:** Teams that need ultra-fast, memory-safe inference on lean servers or edge devices without the overhead of the Python runtime.

---

## 🛠️ Core Features

- **Macro Magic (`#[derive(Module)]`):** Define neural networks just like PyTorch. The macro automatically handles parameter registration and gradient tracking.
- **Dynamic Autograd Engine:** A fully dynamic, eager-execution, reverse-mode automatic differentiation engine.
- **Native Serialization:** Powered by HuggingFace's `safetensors`, models save and load with zero-copy overhead and maximum security (no arbitrary code execution like Python's pickle).
- **Extensible Optimizers & Loss:** Built-in `SGD` optimizer and `MSE` loss, easily expandable to custom implementations.

---

## 🚀 How to Use It

### 1. Define a Model
Use the `#[derive(Module)]` macro to automatically register your network's trainable parameters (weights and biases).

```rust
use nexus_ml::{Tensor, Module};
use nexus_ml::nn::Linear;

#[derive(Module)]
pub struct InferenceModel {
    pub fc1: Linear,
    pub fc2: Linear,
}

impl InferenceModel {
    pub fn new() -> Self {
        InferenceModel {
            fc1: Linear::new(10, 5, true), // in_features, out_features, bias
            fc2: Linear::new(5, 2, true),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x1 = self.fc1.forward(x);
        let x2 = x1.relu(); // Built-in activation functions
        self.fc2.forward(&x2)
    }
}
```

### 2. Training Loop
The training loop feels instantly familiar to anyone who has used PyTorch. Just calculate the loss, run `.backward()`, and step the optimizer!

```rust
use nexus_ml::optim::{Optimizer, SGD};
use nexus_ml::loss::mse_loss;

let model = InferenceModel::new();
let mut optimizer = SGD::new(model.parameters(), 0.01); // Learning rate: 0.01

for epoch in 0..100 {
    optimizer.zero_grad();
    
    // Forward pass
    let pred = model.forward(&input_tensor);
    
    // Calculate loss
    let loss = mse_loss(&pred, &target_tensor);
    
    // Autograd Magic
    loss.backward();
    
    // Update weights
    optimizer.step();
}
```

### 3. Save & Load for Production (Safetensors)
Instantly save your trained weights securely and load them into a production binary.

```rust
use nexus_ml::io::{save_model, load_model, StatefulModule};

// Save to disk
save_model(model.state_dict(), "production_model.safetensors").unwrap();

// --- On your production server ---
let mut prod_model = InferenceModel::new();
let weights = load_model("production_model.safetensors").unwrap();
prod_model.load_state_dict(weights);
```

---

## 🌱 Open for Future Improvements

Nexus ML is built with a solid foundation, but there is immense potential for future expansion. Contributions, PRs, and ideas are highly welcome!

**Current Roadmap / Areas for Contribution:**
1. **GPU Acceleration:** Implement compute backends using `WGPU` (for cross-platform & WebAssembly) or `cudarc` (for native NVIDIA acceleration).
2. **Advanced Layers:** Add implementations for `Conv2d`, `BatchNorm`, `LayerNorm`, and `MultiHeadAttention` (Transformers).
3. **Advanced Optimizers:** Implement `Adam` and `AdamW`.
4. **ONNX Interoperability:** Build a utility to load existing PyTorch ONNX models directly into Nexus ML structures.
5. **Data Loaders:** Build a multi-threaded, asynchronous `DataLoader` trait to prevent CPU bottlenecking during training.

---
*Built with ❤️ in Rust.*
