use ndarray::array;
use nexus_ml::Tensor;

#[test]
fn shared_operand() {
    let x = Tensor::new(array![2.0, -3.0].into_dyn(), true);
    x.mul(&x).backward();
    assert_eq!(x.grad().unwrap(), array![4.0, -6.0].into_dyn());
}

#[test]
fn gradient_snapshot() {
    let x = Tensor::new(array![2.0].into_dyn(), true);
    x.mul(&x).backward();
    let mut copy = x.grad().unwrap();
    copy[[0]] = 99.0;
    assert_eq!(x.grad().unwrap()[[0]], 4.0);
}

#[test]
fn data_snapshot() {
    let x = Tensor::new(array![2.0].into_dyn(), true);
    let mut copy = x.data();
    copy[[0]] = 99.0;
    assert_eq!(x.data()[[0]], 2.0);
}
