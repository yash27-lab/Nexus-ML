//! Regression test for the broadcast-gradient bug.
//!
//! Before the fix, a `[1, out]` bias receiving a `[batch, out]` gradient was not
//! reduced over the batch axis, so `optimizer.step()` silently corrupted the bias
//! shape to `[batch, out]` and training with batch size > 1 was broken. These
//! tests only pass once gradients are correctly reduced to the parameter shape.

use ndarray::Array;
use nexus_ml::loss::mse_loss;
use nexus_ml::nn::{Linear, Module as _};
use nexus_ml::optim::{Optimizer, SGD};
use nexus_ml::Tensor;

#[test]
fn bias_shape_is_preserved_with_batch_gt_1() {
    let lin = Linear::new(2, 1, true);
    let mut opt = SGD::new(lin.parameters(), 0.01);

    let x = Tensor::new(
        Array::from_shape_vec((3, 2), vec![1., 2., 3., 4., 5., 6.])
            .unwrap()
            .into_dyn(),
        false,
    );
    let y = Tensor::new(
        Array::from_shape_vec((3, 1), vec![1., 2., 3.])
            .unwrap()
            .into_dyn(),
        false,
    );

    opt.zero_grad();
    let loss = mse_loss(&lin.forward(&x), &y);
    loss.backward();

    let bias = lin.bias.as_ref().unwrap();
    assert_eq!(
        bias.grad().unwrap().shape(),
        &[1, 1],
        "bias grad not reduced over batch"
    );
    opt.step();
    assert_eq!(
        bias.data().shape(),
        &[1, 1],
        "bias shape corrupted by step()"
    );
}

#[test]
fn batch_training_reduces_loss() {
    // y = 2*x0 - x1 + 0.5, learned over a batch of 8 in one forward pass.
    let lin = Linear::new(2, 1, true);
    let mut opt = SGD::new(lin.parameters(), 0.02);

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

    let first = mse_loss(&lin.forward(&x), &y).data()[[0]];
    for _ in 0..500 {
        opt.zero_grad();
        let loss = mse_loss(&lin.forward(&x), &y);
        loss.backward();
        opt.step();
    }
    let last = mse_loss(&lin.forward(&x), &y).data()[[0]];
    assert!(
        last < first * 1e-3,
        "loss did not converge: {first} -> {last}"
    );
}
