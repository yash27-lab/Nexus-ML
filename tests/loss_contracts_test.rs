use ndarray::array;
use nexus_ml::loss::mse_loss;
use nexus_ml::Tensor;

#[test]
fn broadcast_gradient() {
    let p = Tensor::new(ndarray::arr0(2.0).into_dyn(), true);
    let t = Tensor::new(array![0.0, 1.0].into_dyn(), false);
    let loss = mse_loss(&p, &t);
    assert_eq!(loss.data()[[0]], 2.5);
    loss.backward();
    let g = p.grad().unwrap();
    assert_eq!(g.ndim(), 0);
    assert_eq!(g[[]], 3.0);
}
