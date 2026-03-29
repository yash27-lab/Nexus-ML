use crate::Tensor;
use ndarray::Array;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub trait Module {
    fn parameters(&self) -> Vec<Tensor>;
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // Initialize with random uniform for now
        let weight_data = Array::random((in_features, out_features), Uniform::new(-0.1, 0.1)).into_dyn();
        let weight = Tensor::new(weight_data, true);
        
        let bias_tensor = if bias {
            let bias_data = Array::zeros((1, out_features)).into_dyn();
            Some(Tensor::new(bias_data, true))
        } else {
            None
        };
        
        Linear {
            weight,
            bias: bias_tensor,
        }
    }
    
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.matmul(&self.weight);
        if let Some(b) = &self.bias {
            out = out.add(b);
        }
        out
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}
