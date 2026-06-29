//! Gradient-correctness tests: every differentiable op is checked against
//! central finite differences. Run with `--nocapture` to see the deviations.

use ndarray::Array;
use nexus_ml::gradcheck::assert_grad_check;
use nexus_ml::Tensor;

fn t(shape: (usize, usize), vals: Vec<f32>) -> Tensor {
    Tensor::new(Array::from_shape_vec(shape, vals).unwrap().into_dyn(), true)
}

#[test]
fn gradcheck_add_broadcast() {
    // [2,3] + [1,3] exercises the broadcast-gradient reduction (the bias bug).
    let a = t((2, 3), vec![0.5, -0.3, 1.1, 0.2, -0.8, 0.9]);
    let b = t((1, 3), vec![0.4, -0.6, 0.7]);
    assert_grad_check("add_broadcast", &[a, b], |x| x[0].add(&x[1]));
}

#[test]
fn gradcheck_sub() {
    let a = t((2, 2), vec![0.5, -0.3, 1.1, 0.2]);
    let b = t((2, 2), vec![0.1, 0.9, -0.4, 0.7]);
    assert_grad_check("sub", &[a, b], |x| x[0].sub(&x[1]));
}

#[test]
fn gradcheck_mul() {
    let a = t((2, 2), vec![0.5, -0.3, 1.1, 0.2]);
    let b = t((2, 2), vec![0.1, 0.9, -0.4, 0.7]);
    assert_grad_check("mul", &[a, b], |x| x[0].mul(&x[1]));
}

#[test]
fn gradcheck_matmul() {
    let a = t((2, 3), vec![0.5, -0.3, 1.1, 0.2, -0.8, 0.9]);
    let b = t((3, 2), vec![0.4, -0.6, 0.7, 0.1, -0.2, 0.3]);
    assert_grad_check("matmul", &[a, b], |x| x[0].matmul(&x[1]));
}

#[test]
fn gradcheck_relu() {
    // Values kept away from 0 (relu is non-differentiable at the kink).
    let a = t((2, 3), vec![0.5, -0.7, 1.2, -2.0, 0.3, -1.1]);
    assert_grad_check("relu", &[a], |x| x[0].relu());
}

#[test]
fn gradcheck_sigmoid() {
    let a = t((2, 3), vec![0.5, -0.7, 1.2, -2.0, 0.3, -1.1]);
    assert_grad_check("sigmoid", &[a], |x| x[0].sigmoid());
}

#[test]
fn gradcheck_tanh() {
    let a = t((2, 3), vec![0.5, -0.7, 1.2, -2.0, 0.3, -1.1]);
    assert_grad_check("tanh", &[a], |x| x[0].tanh());
}

#[test]
fn gradcheck_sum() {
    let a = t((2, 3), vec![0.5, -0.7, 1.2, -2.0, 0.3, -1.1]);
    assert_grad_check("sum", &[a], |x| x[0].sum());
}

#[test]
fn gradcheck_mean() {
    let a = t((2, 3), vec![0.5, -0.7, 1.2, -2.0, 0.3, -1.1]);
    assert_grad_check("mean", &[a], |x| x[0].mean());
}

#[test]
fn gradcheck_chain_matmul_tanh_sigmoid() {
    // Composite graph through several ops. Uses smooth activations only so the
    // finite-difference check isn't confounded by ReLU's kink at 0.
    let a = t((2, 3), vec![0.5, -0.3, 1.1, 0.2, -0.8, 0.9]);
    let w = t((3, 2), vec![0.4, -0.6, 0.7, 0.1, -0.2, 0.3]);
    assert_grad_check("chain", &[a, w], |x| x[0].matmul(&x[1]).tanh().sigmoid());
}
