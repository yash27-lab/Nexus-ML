use crate::Tensor;
use ndarray::Array;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD {
    parameters: Vec<Tensor>,
    lr: f32,
}

impl SGD {
    pub fn new(parameters: Vec<Tensor>, lr: f32) -> Self {
        SGD { parameters, lr }
    }
}

impl Optimizer for SGD {
    fn step(&mut self) {
        for param in &self.parameters {
            let mut node = param.node.borrow_mut();
            if let Some(ref grad) = node.grad {
                let grad_update = grad * self.lr;
                node.data = &node.data - &grad_update;
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
}

pub struct AdamW {
    parameters: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: i32,
    m: Vec<ndarray::ArrayD<f32>>,
    v: Vec<ndarray::ArrayD<f32>>,
}

impl AdamW {
    pub fn new(parameters: Vec<Tensor>, lr: f32) -> Self {
        let mut m = Vec::new();
        let mut v = Vec::new();
        for p in &parameters {
            let shape = p.data().shape().to_vec();
            m.push(Array::zeros(shape.clone()).into_dyn());
            v.push(Array::zeros(shape).into_dyn());
        }
        
        AdamW {
            parameters,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            t: 0,
            m,
            v,
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self) {
        self.t += 1;
        
        for (i, param) in self.parameters.iter().enumerate() {
            let mut node = param.node.borrow_mut();
            if let Some(ref grad) = node.grad {
                // Weight decay
                let mut p_data = node.data.clone();
                p_data = &p_data - &(&p_data * (self.lr * self.weight_decay));
                
                // Update biased first moment estimate
                self.m[i] = &self.m[i] * self.beta1 + grad * (1.0 - self.beta1);
                
                // Update biased second raw moment estimate
                let grad_squared = grad * grad;
                self.v[i] = &self.v[i] * self.beta2 + &grad_squared * (1.0 - self.beta2);
                
                // Compute bias-corrected first moment estimate
                let bias_correction1 = 1.0 - self.beta1.powi(self.t);
                let m_hat = &self.m[i] / bias_correction1;
                
                // Compute bias-corrected second raw moment estimate
                let bias_correction2 = 1.0 - self.beta2.powi(self.t);
                let v_hat = &self.v[i] / bias_correction2;
                
                // Update parameters
                let denom = v_hat.mapv(|x| x.sqrt() + self.eps);
                let update = m_hat / denom;
                
                node.data = p_data - (update * self.lr);
            }
        }
    }

    fn zero_grad(&mut self) {
        for param in &self.parameters {
            param.zero_grad();
        }
    }
}
