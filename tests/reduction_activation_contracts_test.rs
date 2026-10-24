use ndarray::{array, ArrayD};
use nexus_ml::Tensor;

#[test]
fn sigmoid_grad() {
    let x = tensor(array![0.0].into_dyn());
    x.sigmoid().backward();
    close(x.grad().unwrap()[[0]], 0.25);
}

#[test]
fn sigmoid_zero() {
    let x = tensor(array![0.0].into_dyn());
    close(x.sigmoid().data()[[0]], 0.5);
}

#[test]
fn sub_broadcast() {
    let x = tensor(array![1.0, 2.0, 3.0].into_dyn());
    let s = tensor(ndarray::arr0(2.0).into_dyn());
    x.sub(&s).sum().backward();
    close(s.grad().unwrap()[[]], -3.0);
}

#[test]
fn sub_right() {
    let a = tensor(array![3.0, 4.0].into_dyn());
    let b = tensor(array![1.0, 2.0].into_dyn());
    a.sub(&b).backward();
    assert_eq!(b.grad().unwrap(), array![-1.0, -1.0].into_dyn());
    assert_eq!(a.grad().unwrap(), array![1.0, 1.0].into_dyn());
}

#[test]
fn sub_self() {
    let x = tensor(array![2.0, 3.0].into_dyn());
    let y = x.sub(&x);
    assert_eq!(y.data(), array![0.0, 0.0].into_dyn());
    y.backward();
    assert_eq!(x.grad().unwrap(), array![0.0, 0.0].into_dyn());
}

#[test]
fn mean_scaled() {
    let x = tensor(array![1.0, 2.0].into_dyn());
    let scale = Tensor::new(array![4.0].into_dyn(), false);
    x.mean().mul(&scale).backward();
    assert_eq!(x.grad().unwrap(), array![2.0, 2.0].into_dyn());
}

#[test]
fn mean_singleton() {
    let x = tensor(array![7.0].into_dyn());
    let y = x.mean();
    close(y.data()[[0]], 7.0);
    y.backward();
    close(x.grad().unwrap()[[0]], 1.0);
}

#[test]
fn mean_grad() {
    let x = tensor(array![2.0, 4.0, 6.0, 8.0].into_dyn());
    x.mean().backward();
    assert_eq!(x.grad().unwrap(), array![0.25, 0.25, 0.25, 0.25].into_dyn());
}

#[test]
fn mean_value() {
    let x = tensor(array![2.0, 4.0, 6.0, 8.0].into_dyn());
    let y = x.mean();
    assert_eq!(y.data().shape(), &[1]);
    close(y.data()[[0]], 5.0);
}

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
