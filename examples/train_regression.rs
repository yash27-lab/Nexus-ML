//! End-to-end training demo: fit `y = 2*x0 - x1 + 0.5` with a single Linear layer
//! over a *batch* of examples in one forward pass.
//!
//!     cargo run --release --example train_regression
//!
//! This exercises the full stack — autograd, broadcasting bias, MSE loss, SGD —
//! and prints the loss curve plus the recovered weights (which should approach
//! the true [2, -1] and bias 0.5).

use ndarray::Array;
use nexus_ml::loss::mse_loss;
use nexus_ml::nn::{Linear, Module as _};
use nexus_ml::optim::{Optimizer, SGD};
use nexus_ml::Tensor;

fn main() {
    let model = Linear::new(2, 1, true);
    let mut opt = SGD::new(model.parameters(), 0.02);

    // Batch of 8 (x0, x1) pairs.
    let xs: Vec<f32> = vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, -1.0, 2.0, 0.5, 0.5, 2.0, -1.0, -2.0, 0.0,
    ];
    let x = Tensor::new(
        Array::from_shape_vec((8, 2), xs.clone())
            .unwrap()
            .into_dyn(),
        false,
    );
    let targets: Vec<f32> = (0..8)
        .map(|i| 2.0 * xs[i * 2] - xs[i * 2 + 1] + 0.5)
        .collect();
    let y = Tensor::new(
        Array::from_shape_vec((8, 1), targets).unwrap().into_dyn(),
        false,
    );

    println!("training  y = 2*x0 - x1 + 0.5  (batch of 8)\n");
    for epoch in 0..=500 {
        opt.zero_grad();
        let loss = mse_loss(&model.forward(&x), &y);
        loss.backward();
        opt.step();
        if epoch % 100 == 0 {
            println!("  epoch {epoch:>3}   loss = {:.6}", loss.data()[[0]]);
        }
    }

    let w = model.weight.data();
    let b = model.bias.as_ref().unwrap().data();
    println!(
        "\nlearned  w0={:.3} w1={:.3}  b={:.3}   (true: 2.000, -1.000, 0.500)",
        w[[0, 0]],
        w[[1, 0]],
        b[[0, 0]]
    );
}
