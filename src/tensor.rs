use crate::autograd::Node;
use ndarray::ArrayD;
use std::cell::RefCell;
use std::rc::Rc;

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
            // Capture operand shapes so broadcast gradients can be reduced back.
            let self_shape = self_data.shape().to_vec();
            let other_shape = other_data.shape().to_vec();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    let g = crate::autograd::reduce_grad_to_shape(grad.clone(), &self_shape);
                    self_clone.node.borrow_mut().add_grad(g);
                }
                if other_clone.node.borrow().requires_grad {
                    let g = crate::autograd::reduce_grad_to_shape(grad.clone(), &other_shape);
                    other_clone.node.borrow_mut().add_grad(g);
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
            let self_shape = self_data.shape().to_vec();
            let other_shape = other_data.shape().to_vec();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    let grad_self = grad * &other_data_clone;
                    let g = crate::autograd::reduce_grad_to_shape(grad_self, &self_shape);
                    self_clone.node.borrow_mut().add_grad(g);
                }
                if other_clone.node.borrow().requires_grad {
                    let grad_other = grad * &self_data_clone;
                    let g = crate::autograd::reduce_grad_to_shape(grad_other, &other_shape);
                    other_clone.node.borrow_mut().add_grad(g);
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
                    let grad_self =
                        grad * self_data_clone.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
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

        let self_2d = self_data
            .into_dimensionality::<ndarray::Ix2>()
            .expect("matmul requires 2D tensors");
        let other_2d = other_data
            .into_dimensionality::<ndarray::Ix2>()
            .expect("matmul requires 2D tensors");

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

    /// Element-wise subtraction `self - other` (with broadcasting).
    pub fn sub(&self, other: &Tensor) -> Tensor {
        let self_data = self.data();
        let other_data = other.data();
        let result_data = &self_data - &other_data;
        let requires_grad = self.node.borrow().requires_grad || other.node.borrow().requires_grad;
        let mut result_node = Node::new(result_data, requires_grad);

        if requires_grad {
            let self_clone = self.clone();
            let other_clone = other.clone();
            let self_shape = self_data.shape().to_vec();
            let other_shape = other_data.shape().to_vec();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    let g = crate::autograd::reduce_grad_to_shape(grad.clone(), &self_shape);
                    self_clone.node.borrow_mut().add_grad(g);
                }
                if other_clone.node.borrow().requires_grad {
                    let g = crate::autograd::reduce_grad_to_shape(-grad.clone(), &other_shape);
                    other_clone.node.borrow_mut().add_grad(g);
                }
            }));
            result_node.parents = vec![self.clone(), other.clone()];
        }
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// Sum of all elements, returning a scalar (shape `[1]`) tensor.
    pub fn sum(&self) -> Tensor {
        let self_data = self.data();
        let total = self_data.sum();
        let result_data = ndarray::Array::from_elem((1,), total).into_dyn();
        let requires_grad = self.node.borrow().requires_grad;
        let mut result_node = Node::new(result_data, requires_grad);

        if requires_grad {
            let self_clone = self.clone();
            let self_shape = self_data.shape().to_vec();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    // d(sum)/d(x_i) = 1, so each element gets the upstream scalar.
                    let g = ArrayD::from_elem(self_shape.clone(), grad[[0]]);
                    self_clone.node.borrow_mut().add_grad(g);
                }
            }));
            result_node.parents = vec![self.clone()];
        }
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// Mean of all elements, returning a scalar (shape `[1]`) tensor.
    pub fn mean(&self) -> Tensor {
        let self_data = self.data();
        let n = self_data.len() as f32;
        let result_data = ndarray::Array::from_elem((1,), self_data.sum() / n).into_dyn();
        let requires_grad = self.node.borrow().requires_grad;
        let mut result_node = Node::new(result_data, requires_grad);

        if requires_grad {
            let self_clone = self.clone();
            let self_shape = self_data.shape().to_vec();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    let g = ArrayD::from_elem(self_shape.clone(), grad[[0]] / n);
                    self_clone.node.borrow_mut().add_grad(g);
                }
            }));
            result_node.parents = vec![self.clone()];
        }
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// Sigmoid activation `1 / (1 + e^-x)`.
    pub fn sigmoid(&self) -> Tensor {
        let s = self.data().mapv(|x| 1.0 / (1.0 + (-x).exp()));
        let requires_grad = self.node.borrow().requires_grad;
        let mut result_node = Node::new(s.clone(), requires_grad);

        if requires_grad {
            let self_clone = self.clone();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    // d(sigmoid)/dx = s * (1 - s)
                    let local = &s * &s.mapv(|v| 1.0 - v);
                    self_clone.node.borrow_mut().add_grad(grad * &local);
                }
            }));
            result_node.parents = vec![self.clone()];
        }
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }

    /// Hyperbolic tangent activation.
    pub fn tanh(&self) -> Tensor {
        let t = self.data().mapv(|x| x.tanh());
        let requires_grad = self.node.borrow().requires_grad;
        let mut result_node = Node::new(t.clone(), requires_grad);

        if requires_grad {
            let self_clone = self.clone();
            result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
                if self_clone.node.borrow().requires_grad {
                    // d(tanh)/dx = 1 - tanh^2
                    let local = t.mapv(|v| 1.0 - v * v);
                    self_clone.node.borrow_mut().add_grad(grad * &local);
                }
            }));
            result_node.parents = vec![self.clone()];
        }
        Tensor {
            node: Rc::new(RefCell::new(result_node)),
        }
    }
}
