<div align="center">
  <h1>Nexus ML</h1>
  <p><strong>A small, correctness-first deep-learning framework in pure Rust.</strong></p>

[![CI](https://github.com/yash27-lab/Nexus-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/yash27-lab/Nexus-ML/actions/workflows/ci.yml)
</div>

---

## What it is

Nexus ML is a from-scratch, eager-execution **reverse-mode autograd engine** with a
PyTorch-like API, written in Rust. You define a model with a derive macro, write a
normal training loop, call `.backward()`, and step an optimizer — no Python runtime,
no FFI.

The thing that makes it trustworthy is that **every differentiable operation is
verified against finite-difference gradients** (`src/gradcheck.rs`). An autograd
engine lives or dies by whether its gradients are correct; here that's a test you
can run, not a claim.

```bash
cargo test --test gradcheck_test -- --nocapture
```
```
[gradcheck] add_broadcast  max|Δ| = 1.45e-4
[gradcheck] matmul         max|Δ| = 5.57e-5
[gradcheck] sigmoid        max|Δ| = ...
... every op, analytical vs numerical
```

## It actually learns

```bash
cargo run --release --example train_regression
```
```
training  y = 2*x0 - x1 + 0.5  (batch of 8)
  epoch   0   loss = 9.606565
  epoch 100   loss = 0.005837
  epoch 500   loss = 0.000000
learned  w0=2.000 w1=-1.000  b=0.500   (true: 2.000, -1.000, 0.500)
```

## A real bug the gradient checker caught

This is why the verification matters. The original `add` backward passed the upstream
gradient straight to both operands. When a `[1, out]` bias was broadcast against a
`[batch, out]` activation, the bias received a `[batch, out]` gradient that was never
reduced over the batch axis — so `optimizer.step()` **silently corrupted the bias to
`[batch, out]` and training with batch size > 1 was broken.** The existing tests all
used batch size 1, so they never hit it.

The fix reduces each gradient back to its operand's shape (summing over broadcast
axes). It's now locked in by `tests/batch_training_test.rs`, which trains with a real
batch and asserts both convergence and that parameter shapes are preserved.

## Verified operations

Each has an analytical backward checked against finite differences:

| Category | Ops |
|----------|-----|
| Arithmetic | `add`, `sub`, `mul` (all broadcasting-aware), `matmul` |
| Activations | `relu`, `sigmoid`, `tanh` |
| Reductions | `sum`, `mean` |
| Layers | `Linear` (weight + optional bias) |
| Loss | `mse_loss` |
| Optimizers | `SGD`, `AdamW` (with bias correction + decoupled weight decay) |

## Usage

```rust
use nexus_ml::{Tensor, Module};
use nexus_ml::nn::Linear;
use nexus_ml::optim::{Optimizer, SGD};
use nexus_ml::loss::mse_loss;

#[derive(Module)]                       // auto-registers trainable params
struct Net { fc1: Linear, fc2: Linear }

impl Net {
    fn new() -> Self { Net { fc1: Linear::new(10, 5, true), fc2: Linear::new(5, 2, true) } }
    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc2.forward(&self.fc1.forward(x).relu())
    }
}

let model = Net::new();
let mut opt = SGD::new(model.parameters(), 0.01);
for _ in 0..100 {
    opt.zero_grad();
    let loss = mse_loss(&model.forward(&input), &target);
    loss.backward();
    opt.step();
}
```

Save/load weights with `safetensors` (little-endian, spec-compliant, no pickle):

```rust
use nexus_ml::io::{save_model, load_model, StatefulModule};
save_model(model.state_dict(), "model.safetensors")?;
let weights = load_model("model.safetensors")?;
prod_model.load_state_dict(weights);
```

## How the autograd works

- `Tensor` is an `Rc<RefCell<Node>>` into a dynamic graph. Each op records a
  `backward_fn` closure and its parents.
- `backward()` does a topological sort from the root, seeds the root gradient with
  ones, and walks nodes in reverse, accumulating gradients into each parameter.
- Broadcasting is handled by `reduce_grad_to_shape`, which sums an upstream gradient
  back to its operand's shape.

```
src/tensor.rs     ops + their backward closures
src/autograd.rs   Node, topological backward, gradient reduction
src/gradcheck.rs  finite-difference gradient verification
src/nn.rs         Module trait + Linear
src/optim.rs      SGD, AdamW
macros/           #[derive(Module)] parameter registration
```

## Status & roadmap

Implemented and verified: the items in the table above, the derive macro, safetensors
I/O, and end-to-end batched training. This is a focused engine, **not** a PyTorch
replacement. Honest next steps:

1. **Batched/broadcasting `matmul`** (currently 2-D only) and `Conv2d` / `LayerNorm` /
   attention layers.
2. **GPU backend** via `wgpu` or `cudarc`.
3. **`#[derive(StatefulModule)]`** to generate `state_dict` automatically (it's manual today).
4. Broader op coverage (softmax, cross-entropy) and a multi-threaded `DataLoader`.

Contributions welcome — every new op must ship with a `gradcheck` test.

## License

See [LICENSE](LICENSE).
