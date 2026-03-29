use nexus_ml::{Tensor, Module};
use nexus_ml::nn::{Linear, Module as NnModule};
use nexus_ml::optim::{Optimizer, SGD};
use nexus_ml::loss::mse_loss;
use ndarray::Array;

#[derive(Module)]
struct TinyModel {
    fc1: Linear,
}

impl TinyModel {
    fn new() -> Self {
        TinyModel {
            fc1: Linear::new(1, 1, true),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        self.fc1.forward(x)
    }
}

#[test]
fn test_training_loop() {
    let model = TinyModel::new();
    let mut optimizer = SGD::new(model.parameters(), 0.01);

    // Simple dataset: y = 2x + 1
    let x_data = vec![
        Array::from_shape_vec((1, 1), vec![1.0]).unwrap().into_dyn(),
        Array::from_shape_vec((1, 1), vec![2.0]).unwrap().into_dyn(),
        Array::from_shape_vec((1, 1), vec![3.0]).unwrap().into_dyn(),
    ];
    
    let y_data = vec![
        Array::from_shape_vec((1, 1), vec![3.0]).unwrap().into_dyn(),
        Array::from_shape_vec((1, 1), vec![5.0]).unwrap().into_dyn(),
        Array::from_shape_vec((1, 1), vec![7.0]).unwrap().into_dyn(),
    ];

    let mut initial_loss = 0.0;
    let mut final_loss = 0.0;

    for epoch in 0..100 {
        let mut epoch_loss = 0.0;
        
        for (i, x) in x_data.iter().enumerate() {
            let x_tensor = Tensor::new(x.clone(), false);
            let y_tensor = Tensor::new(y_data[i].clone(), false);

            optimizer.zero_grad();
            
            let pred = model.forward(&x_tensor);
            let loss = mse_loss(&pred, &y_tensor);
            
            loss.backward();
            optimizer.step();

            epoch_loss += loss.data()[[0]];
        }
        
        if epoch == 0 {
            initial_loss = epoch_loss;
        }
        if epoch == 99 {
            final_loss = epoch_loss;
        }
    }

    // The loss should have significantly decreased after 100 epochs
    assert!(final_loss < initial_loss);
    println!("Initial Loss: {}, Final Loss: {}", initial_loss, final_loss);
}
