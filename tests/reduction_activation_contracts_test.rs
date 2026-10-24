use ndarray::{array, ArrayD};
use nexus_ml::Tensor;

#[test]
fn sum_scaled() {
    let x = tensor(array![1.0, 2.0].into_dyn());
    let scale = Tensor::new(array![3.0].into_dyn(), false);
    x.sum().mul(&scale).backward();
    assert_eq!(x.grad().unwrap(), array![3.0, 3.0].into_dyn());
}

#[test]
fn sum_matrix() {
    let x = tensor(array![[1.0, 2.0], [3.0, 4.0]].into_dyn());
    x.sum().backward();
    assert_eq!(x.grad().unwrap().shape(), &[2, 2]);
    assert!(x.grad().unwrap().iter().all(|v| *v == 1.0));
}

#[test]
fn sum_grad() {
    let x = tensor(array![1.0, 2.0, 3.0].into_dyn());
    x.sum().backward();
    assert_eq!(x.grad().unwrap(), array![1.0, 1.0, 1.0].into_dyn());
}

#[test]
fn sum_value() {
    let x = tensor(array![1.0, -2.0, 4.0].into_dyn());
    let y = x.sum();
    assert_eq!(y.data().shape(), &[1]);
    close(y.data()[[0]], 3.0);
}

fn tensor(data: ArrayD<f32>) -> Tensor {
    Tensor::new(data, true)
}

fn close(actual: f32, expected: f32) {
    assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
}
