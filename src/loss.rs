use crate::Tensor;
use ndarray::ArrayD;

/// Mean Squared Error Loss
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    let pred_data = predictions.data();
    let target_data = targets.data();
    
    // (pred - target)^2
    let diff = &pred_data - &target_data;
    let squared_diff = &diff * &diff;
    
    // Mean over all elements
    let n_elements = squared_diff.len() as f32;
    let sum = squared_diff.sum();
    let mean_loss = sum / n_elements;
    
    let result_data = ndarray::Array::from_elem((1,), mean_loss).into_dyn();
    let requires_grad = predictions.node.borrow().requires_grad;
    
    let mut result_node = crate::autograd::Node::new(result_data, requires_grad);
    
    if requires_grad {
        let pred_clone = predictions.clone();
        
        result_node.backward_fn = Some(Box::new(move |grad: &ArrayD<f32>| {
            if pred_clone.node.borrow().requires_grad {
                let upstream_grad = grad[[0]]; // scalar gradient from upstream
                // d(MSE)/d(pred) = 2 * (pred - target) / N
                let grad_pred = (&diff * (2.0 / n_elements)) * upstream_grad;
                pred_clone.node.borrow_mut().add_grad(grad_pred);
            }
        }));
        result_node.parents = vec![predictions.clone()];
    }
    
    Tensor {
        node: std::rc::Rc::new(std::cell::RefCell::new(result_node)),
    }
}
