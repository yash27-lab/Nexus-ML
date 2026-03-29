# Nexus ML 🚀

A Pure Rust Machine Learning Framework built to eliminate the **Research-to-Production gap**.

## The Vision
In the current ecosystem, data scientists design and train models in Python (PyTorch/TensorFlow), but deploying these models at scale in production often requires rewriting inference code in C++, exporting to ONNX, or dealing with the operational nightmare of scaling Python environments.

**Nexus ML** changes this by providing a unified, high-performance library written entirely in Rust. It pairs the expressiveness and ergonomics of PyTorch's model definition (using Rust's powerful procedural macros) with the raw speed and zero-cost abstraction deployment of a compiled binary.

## Features

### ✨ Macro Magic for Model Definition
Defining neural networks feels like PyTorch, without sacrificing Rust's strict typing. The `#[derive(Module)]` macro automatically implements parameter registration and autograd tracking for all your layers.

```rust
use nexus_ml::{Tensor, Module};
use nexus_ml::nn::Linear;

#[derive(Module)]
struct MyModel {
    fc1: Linear,
    fc2: Linear,
}

impl MyModel {
    fn new() -> Self {
        MyModel {
            fc1: Linear::new(10, 5, true),
            fc2: Linear::new(5, 1, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x1 = self.fc1.forward(x);
        let x2 = x1.relu();
        self.fc2.forward(&x2)
    }
}
```

### 🧠 Dynamic Autograd Engine
A fully dynamic, eager-execution, reverse-mode automatic differentiation engine. Call `.backward()` on your loss tensor, and gradients flow seamlessly back through `relu`, `matmul`, `add`, and more.

```rust
let pred = model.forward(&input);
let loss = mse_loss(&pred, &target);
loss.backward();
```

### ⚡ Fast & Safe Serialization
Powered by HuggingFace's `safetensors`, you can instantly save trained weights and load them straight into a production inference binary. No pickling. No arbitrary code execution. Zero-copy loading.

```rust
use nexus_ml::io::{save_model, load_model, StatefulModule};

// Save
save_model(model.state_dict(), "model.safetensors").unwrap();

// Load in Production
let state_dict = load_model("model.safetensors").unwrap();
model.load_state_dict(state_dict);
```

### 🏃 Optimizers and Training
Includes standard optimizers like SGD out of the box to loop through data and apply gradients easily.

```rust
let mut optimizer = SGD::new(model.parameters(), 0.01);
optimizer.zero_grad();
// ... forward pass ...
loss.backward();
optimizer.step();
```

## Running the Examples
You can run the full test suite to see the framework train a model from scratch and correctly serialize it:

```bash
cargo test
```

## Architecture
1. **`tensor`**: Wrapper over `ndarray` linking data to our dynamic computational graph.
2. **`autograd`**: Manages topological graph construction and backward passes.
3. **`nn`**: Neural network layers (e.g., `Linear`) and the `Module` trait.
4. **`optim`**: Gradient descent algorithms to update weights.
5. **`io`**: Safe, memory-mapped parameter saving and loading.
6. **`macros`**: Procedural macros (`#[derive(Module)]`) for ergonomic DX.

## Future Roadmap
- CUDA / GPU backend integration via WGPU or cudarc.
- Advanced layers (Conv2d, Transformers/Attention, LayerNorm).
- More optimizers (AdamW).
- Cross-platform WebAssembly deployment target.
