use crate::Tensor;

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
                // Update rule: data = data - lr * grad
                // Extract into variable to avoid borrow checker issues with node.data
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
