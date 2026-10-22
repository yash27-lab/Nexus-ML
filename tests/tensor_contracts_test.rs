use ndarray::array;
use nexus_ml::Tensor;

#[test]
fn data_snapshot() {
    let x = Tensor::new(array![2.0].into_dyn(), true);
    let mut copy = x.data();
    copy[[0]] = 99.0;
    assert_eq!(x.data()[[0]], 2.0);
}
