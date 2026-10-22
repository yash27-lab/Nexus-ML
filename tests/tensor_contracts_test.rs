use ndarray::array;
use nexus_ml::Tensor;

#[test]
fn constant_chain() {
    let x = Tensor::new(array![-1.0, 2.0].into_dyn(), false);
    let y = x.add(&x).relu();
    assert_eq!(y.data(), array![0.0, 4.0].into_dyn());
    assert!(x.grad().is_none());
    assert!(y.grad().is_none());
}

#[test]
fn broadcast_add() {
    let x = Tensor::new(array![1.0, 2.0, 3.0].into_dyn(), true);
    let scalar = Tensor::new(ndarray::arr0(2.0).into_dyn(), true);
    x.add(&scalar).backward();
    let grad = scalar.grad().unwrap();
    assert_eq!(grad.ndim(), 0);
    assert_eq!(grad[[]], 3.0);
}

#[test]
fn clone_alias() {
    let x = Tensor::new(array![2.0].into_dyn(), true);
    let alias = x.clone();
    x.add(&alias).backward();
    assert_eq!(x.grad().unwrap()[[0]], 2.0);
    assert_eq!(alias.grad().unwrap(), x.grad().unwrap());
}

#[test]
fn relu_boundary() {
    let x = Tensor::new(array![-2.0, 0.0, 3.0].into_dyn(), true);
    let y = x.relu();
    assert_eq!(y.data(), array![0.0, 0.0, 3.0].into_dyn());
    y.backward();
    assert_eq!(x.grad().unwrap(), array![0.0, 0.0, 1.0].into_dyn());
}

#[test]
fn constant_operand() {
    let x = Tensor::new(array![2.0].into_dyn(), true);
    let c = Tensor::new(array![3.0].into_dyn(), false);
    x.mul(&c).backward();
    assert_eq!(x.grad().unwrap()[[0]], 3.0);
    assert!(c.grad().is_none());
}

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
