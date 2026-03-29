use nexus_ml::{Tensor, Module};
use nexus_ml::nn::{Linear, Module as NnModule};
use ndarray::Array;

#[derive(Module)]
struct MyModel {
    fc1: Linear,
    fc2: Linear,
}

impl MyModel {
    fn new() -> Self {
        MyModel {
            fc1: Linear::new(2, 4, true),
            fc2: Linear::new(4, 1, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let x1 = self.fc1.forward(x);
        let x2 = x1.relu();
        self.fc2.forward(&x2)
    }
}

#[test]
fn test_nn_macro_and_forward() {
    let model = MyModel::new();
    
    // Check parameters collection from macro
    let params = model.parameters();
    assert_eq!(params.len(), 4); // fc1.weight, fc1.bias, fc2.weight, fc2.bias

    // Run a forward pass
    let input_data = Array::from_shape_vec((1, 2), vec![0.5, -0.5]).unwrap().into_dyn();
    let input = Tensor::new(input_data, false);
    
    let output = model.forward(&input);
    let out_shape = output.data().shape().to_vec();
    assert_eq!(out_shape, vec![1, 1]);
    
    // Run backward pass
    output.backward();
    
    // Check gradients exist
    for param in params {
        assert!(param.grad().is_some());
    }
}
