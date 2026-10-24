use ndarray::{array, ArrayD};
use nexus_ml::Tensor;

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
