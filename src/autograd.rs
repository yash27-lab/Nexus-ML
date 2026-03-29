use std::collections::HashSet;
use std::rc::Rc;
use std::cell::RefCell;
use ndarray::{ArrayD, Array};

use crate::tensor::Tensor;

pub(crate) type BackwardFn = Box<dyn Fn(&ArrayD<f32>)>;

pub(crate) struct Node {
    pub(crate) data: ArrayD<f32>,
    pub(crate) grad: Option<ArrayD<f32>>,
    pub(crate) requires_grad: bool,
    pub(crate) backward_fn: Option<BackwardFn>,
    pub(crate) parents: Vec<Tensor>,
}

impl Node {
    pub(crate) fn new(data: ArrayD<f32>, requires_grad: bool) -> Self {
        Node {
            data,
            grad: None,
            requires_grad,
            backward_fn: None,
            parents: vec![],
        }
    }

    pub(crate) fn zero_grad(&mut self) {
        self.grad = None;
    }

    pub(crate) fn add_grad(&mut self, grad: ArrayD<f32>) {
        if let Some(ref mut current_grad) = self.grad {
            *current_grad = current_grad.clone() + grad;
        } else {
            self.grad = Some(grad);
        }
    }
}

pub(crate) fn backward(root: Tensor) {
    let mut topo = vec![];
    let mut visited = HashSet::new();

    fn build_topo(tensor: &Tensor, topo: &mut Vec<Tensor>, visited: &mut HashSet<*const RefCell<Node>>) {
        let ptr = Rc::as_ptr(&tensor.node);
        if !visited.contains(&ptr) {
            visited.insert(ptr);
            for parent in &tensor.node.borrow().parents {
                build_topo(parent, topo, visited);
            }
            topo.push(tensor.clone());
        }
    }

    build_topo(&root, &mut topo, &mut visited);
    topo.reverse();

    // Set the root gradient to 1.0
    let root_shape = root.node.borrow().data.shape().to_vec();
    let ones = Array::ones(root_shape).into_dyn();
    root.node.borrow_mut().grad = Some(ones);

    for tensor in topo {
        let node = tensor.node.borrow();
        if let Some(ref backward_fn) = node.backward_fn {
            if let Some(ref grad) = node.grad {
                backward_fn(grad);
            }
        }
    }
}
