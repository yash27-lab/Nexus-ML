use std::rc::Rc;
use std::cell::RefCell;
use ndarray::ArrayD;
use crate::autograd::Node;

/// The core Tensor type.
/// Wraps a dynamic computational graph node.
#[derive(Clone)]
pub struct Tensor {
    pub(crate) node: Rc<RefCell<Node>>,
}

impl Tensor {
    /// Create a new Tensor from an ndarray, requiring gradients.
    pub fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Tensor {
            node: Rc::new(RefCell::new(Node::new(data, requires_grad))),
        }
    }

    /// Retrieve a copy of the underlying data.
    pub fn data(&self) -> ArrayD<f32> {
        self.node.borrow().data.clone()
    }

    /// Retrieve a copy of the accumulated gradients.
    pub fn grad(&self) -> Option<ArrayD<f32>> {
        self.node.borrow().grad.clone()
    }

    /// Zero out the gradients.
    pub fn zero_grad(&self) {
        self.node.borrow_mut().zero_grad();
    }

    /// Perform the backward pass starting from this tensor.
    pub fn backward(&self) {
        crate::autograd::backward(self.clone());
    }

    /// Add two tensors together.
    pub fn add(&self, other: &Tensor) -> Tensor {
        let self_data = self.data();
        let other_data = other.data();
        let result_data = &self_data + &other_data;

        let requires_grad = self.node.borrow().requires_grad || other.node.borrow().requires_grad;
        
        let result_node = Node::new(result_data, requires_grad);
        let mut result_node = result_node;
        
        if requires_grad {
            let self_clone = self.clone();
            let other_clone = other.clone();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    self_clone.node.borrow_mut().add_grad(grad.clone());
                }
                if other_clone.node.borrow().requires_grad {
                    other_clone.node.borrow_mut().add_grad(grad.clone());
                }
            }));
            result_node.parents = vec![self.clone(), other.clone()];
        }
        
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// Multiply two tensors (element-wise).
    pub fn mul(&self, other: &Tensor) -> Tensor {
        let self_data = self.data();
        let other_data = other.data();
        let result_data = &self_data * &other_data;

        let requires_grad = self.node.borrow().requires_grad || other.node.borrow().requires_grad;
        
        let result_node = Node::new(result_data, requires_grad);
        let mut result_node = result_node;
        
        if requires_grad {
            let self_clone = self.clone();
            let other_clone = other.clone();
            let self_data_clone = self_data.clone();
            let other_data_clone = other_data.clone();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    let grad_self = grad * &other_data_clone;
                    self_clone.node.borrow_mut().add_grad(grad_self);
                }
                if other_clone.node.borrow().requires_grad {
                    let grad_other = grad * &self_data_clone;
                    other_clone.node.borrow_mut().add_grad(grad_other);
                }
            }));
            result_node.parents = vec![self.clone(), other.clone()];
        }
        
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// ReLU activation function.
    pub fn relu(&self) -> Tensor {
        let self_data = self.data();
        let result_data = self_data.mapv(|x| if x > 0.0 { x } else { 0.0 });
        
        let requires_grad = self.node.borrow().requires_grad;
        let mut result_node = Node::new(result_data, requires_grad);
        
        if requires_grad {
            let self_clone = self.clone();
            let self_data_clone = self_data.clone();
            
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    let grad_self = grad * self_data_clone.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
                    self_clone.node.borrow_mut().add_grad(grad_self);
                }
            }));
            result_node.parents = vec![self.clone()];
        }
        
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// Matrix multiplication (assumes 2D tensors).
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let self_data = self.data();
        let other_data = other.data();
        
        let self_2d = self_data.into_dimensionality::<ndarray::Ix2>().expect("matmul requires 2D tensors");
        let other_2d = other_data.into_dimensionality::<ndarray::Ix2>().expect("matmul requires 2D tensors");
        
        let result_data = self_2d.dot(&other_2d).into_dyn();
        let requires_grad = self.node.borrow().requires_grad || other.node.borrow().requires_grad;
        
        let mut result_node = Node::new(result_data, requires_grad);
        
        if requires_grad {
            let self_clone = self.clone();
            let other_clone = other.clone();
            let self_2d_clone = self_2d.clone();
            let other_2d_clone = other_2d.clone();
            
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                let grad_2d = grad.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
                
                if self_clone.node.borrow().requires_grad {
                    let grad_self = grad_2d.dot(&other_2d_clone.t()).into_dyn();
                    self_clone.node.borrow_mut().add_grad(grad_self);
                }
                if other_clone.node.borrow().requires_grad {
                    let grad_other = self_2d_clone.t().dot(&grad_2d).into_dyn();
                    other_clone.node.borrow_mut().add_grad(grad_other);
                }
            }));
            result_node.parents = vec![self.clone(), other.clone()];
        }
        
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }
}
