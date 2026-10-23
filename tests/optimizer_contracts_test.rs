use ndarray::array;
use nexus_ml::optim::{AdamW, Optimizer, SGD};
use nexus_ml::Tensor;

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
