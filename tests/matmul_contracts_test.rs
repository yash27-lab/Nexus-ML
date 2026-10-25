use ndarray::array;
use nexus_ml::Tensor;

#[test]
fn constant_operands() {
    let a = Tensor::new(array![[1.0, 2.0]].into_dyn(), false);
    let b = Tensor::new(array![[3.0], [4.0]].into_dyn(), false);
    let y = a.matmul(&b);
    assert_eq!(y.data()[[0, 0]], 11.0);
    assert!(y.grad().is_none());
    assert!(a.grad().is_none());
    assert!(b.grad().is_none());
}

#[test]
fn both_gradient_shapes() {
    let a = Tensor::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(), true);
    let b = Tensor::new(array![[1.0], [2.0], [3.0]].into_dyn(), true);
    a.matmul(&b).sum().backward();
    assert_eq!(a.grad().unwrap().shape(), &[2, 3]);
    assert_eq!(b.grad().unwrap().shape(), &[3, 1]);
}

#[test]
fn right_gradient() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), false);
    let b = Tensor::new(array![[2.0], [3.0]].into_dyn(), true);
    a.matmul(&b).sum().backward();
    assert_eq!(b.grad().unwrap(), array![[4.0], [6.0]].into_dyn());
    assert!(a.grad().is_none());
}

#[test]
fn left_gradient() {
    let a = Tensor::new(array![[1.0, 2.0]].into_dyn(), true);
    let b = Tensor::new(array![[3.0], [4.0]].into_dyn(), false);
    a.matmul(&b).sum().backward();
    assert_eq!(a.grad().unwrap(), array![[3.0, 4.0]].into_dyn());
    assert!(b.grad().is_none());
}

#[test]
fn identity_forward() {
    let a = Tensor::new(array![[1.0, 2.0], [3.0, 4.0]].into_dyn(), true);
    let i = Tensor::new(array![[1.0, 0.0], [0.0, 1.0]].into_dyn(), false);
    assert_eq!(a.matmul(&i).data(), a.data());
}

#[test]
fn rectangular_forward() {
    let a = Tensor::new(array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(), true);
    let b = Tensor::new(array![[1.0], [2.0], [3.0]].into_dyn(), true);
    assert_eq!(a.matmul(&b).data(), array![[14.0], [32.0]].into_dyn());
}
