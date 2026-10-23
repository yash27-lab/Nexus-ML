use ndarray::array;
use nexus_ml::optim::{AdamW, Optimizer, SGD};
use nexus_ml::Tensor;

#[test]
fn adam_zero_rate() {
    let x = parameter(2.0);
    x.mul(&x).backward();
    AdamW::new(vec![x.clone()], 0.0).step();
    assert_eq!(x.data()[[0]], 2.0);
}

#[test]
fn adam_no_gradient() {
    let x = parameter(2.0);
    AdamW::new(vec![x.clone()], 0.1).step();
    assert_eq!(x.data()[[0]], 2.0);
}

#[test]
fn adam_empty() {
    let mut o = AdamW::new(vec![], 0.1);
    o.step();
    o.zero_grad();
}

#[test]
fn sgd_unregistered() {
    let x = parameter(2.0);
    let y = parameter(3.0);
    x.mul(&y).backward();
    SGD::new(vec![x], 0.1).step();
    assert_eq!(y.data()[[0]], 3.0);
}

#[test]
fn sgd_shape() {
    let x = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), true);
    x.sum().backward();
    SGD::new(vec![x.clone()], 0.1).step();
    assert_eq!(x.data().shape(), &[2, 2]);
    close(x.data()[[1, 1]], 3.9);
}

#[test]
fn sgd_zero_grad() {
    let x = parameter(2.0);
    x.mul(&x).backward();
    let mut o = SGD::new(vec![x.clone()], 0.1);
    o.zero_grad();
    assert!(x.grad().is_none());
    o.step();
    assert_eq!(x.data()[[0]], 2.0);
}

#[test]
fn sgd_retains_gradient() {
    let x = parameter(2.0);
    x.mul(&x).backward();
    SGD::new(vec![x.clone()], 0.1).step();
    assert_eq!(x.grad().unwrap()[[0]], 4.0);
}

#[test]
fn sgd_negative() {
    let x = parameter(-2.0);
    x.mul(&x).backward();
    SGD::new(vec![x.clone()], 0.1).step();
    close(x.data()[[0]], -1.6);
}

#[test]
fn sgd_positive() {
    let x = parameter(2.0);
    x.mul(&x).backward();
    SGD::new(vec![x.clone()], 0.1).step();
    close(x.data()[[0]], 1.6);
}

#[test]
fn sgd_zero_rate() {
    let x = parameter(2.0);
    x.mul(&x).backward();
    SGD::new(vec![x.clone()], 0.0).step();
    assert_eq!(x.data()[[0]], 2.0);
}

#[test]
fn sgd_no_gradient() {
    let x = parameter(2.0);
    SGD::new(vec![x.clone()], 0.1).step();
    assert_eq!(x.data()[[0]], 2.0);
}

#[test]
fn sgd_empty() {
    let mut o = SGD::new(vec![], 0.1);
    o.step();
    o.zero_grad();
}

fn parameter(value: f32) -> Tensor {
    Tensor::new(array![value].into_dyn(), true)
}

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
}
