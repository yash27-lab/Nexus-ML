use ndarray::Array;
use nexus_ml::Tensor;

#[test]
fn test_basic_autograd() {
    let x_data = Array::from_elem((1,), 2.0).into_dyn();
    let y_data = Array::from_elem((1,), 3.0).into_dyn();

    let x = Tensor::new(x_data, true);
    let y = Tensor::new(y_data, true);

    // z = (x + y) * x
    let sum = x.add(&y);
    let z = sum.mul(&x);

    z.backward();

    let dz_dx = x.grad().unwrap();
    let dz_dy = y.grad().unwrap();

    assert_eq!(dz_dx[[0]], 7.0);
    assert_eq!(dz_dy[[0]], 2.0);
}
