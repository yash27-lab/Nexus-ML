use ndarray::array;
use nexus_ml::Tensor;

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
