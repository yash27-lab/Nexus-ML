//! Finite-difference gradient checking.
//!
//! An autograd engine is only trustworthy if its analytical gradients are
//! actually correct. `grad_check` perturbs each input element by ±eps, measures
//! the change in `sum(f(inputs))` via a central difference, and compares it to
//! the gradient produced by `backward()`. It is the autograd equivalent of a
//! reference-vs-implementation parity test: if numerical and analytical
//! gradients agree, the backward pass is correct by construction of the check.

use ndarray::ArrayD;

use crate::Tensor;

pub struct GradCheckResult {
    pub max_abs_diff: f32,
    pub passed: bool,
}

fn get_flat(t: &Tensor, flat: usize) -> f32 {
    t.node.borrow().data.iter().nth(flat).copied().unwrap()
}

fn set_flat(t: &Tensor, flat: usize, val: f32) {
    let mut node = t.node.borrow_mut();
    if let Some(elem) = node.data.iter_mut().nth(flat) {
        *elem = val;
    }
}

/// Checks the gradients of `f` w.r.t. each tensor in `inputs`.
///
/// `f` maps the inputs to an output tensor; the scalar objective is `sum(out)`,
/// which matches `backward()` seeding the root gradient with ones. Inputs must
/// have `requires_grad = true`. Returns the maximum absolute deviation between
/// numerical and analytical gradients over all input elements.
pub fn grad_check<F>(inputs: &[Tensor], f: F, eps: f32, tol: f32) -> GradCheckResult
where
    F: Fn(&[Tensor]) -> Tensor,
{
    // Analytical gradients from the backward pass.
    for t in inputs {
        t.zero_grad();
    }
    let out = f(inputs);
    out.backward();
    let analytical: Vec<ArrayD<f32>> = inputs
        .iter()
        .map(|t| {
            t.grad()
                .unwrap_or_else(|| ArrayD::zeros(t.data().raw_dim()))
        })
        .collect();

    // Scalar objective for finite differences.
    let objective = |inputs: &[Tensor]| f(inputs).data().sum();

    let mut max_abs_diff = 0.0f32;
    for (ti, t) in inputs.iter().enumerate() {
        let n = t.data().len();
        for flat in 0..n {
            let orig = get_flat(t, flat);
            set_flat(t, flat, orig + eps);
            let plus = objective(inputs);
            set_flat(t, flat, orig - eps);
            let minus = objective(inputs);
            set_flat(t, flat, orig); // restore

            let numerical = (plus - minus) / (2.0 * eps);
            let analytic = analytical[ti].iter().nth(flat).copied().unwrap_or(0.0);
            max_abs_diff = max_abs_diff.max((numerical - analytic).abs());
        }
    }

    GradCheckResult {
        max_abs_diff,
        passed: max_abs_diff <= tol,
    }
}

/// Convenience wrapper for tests: runs `grad_check` with sensible f32 tolerances
/// and panics with a helpful message on failure.
pub fn assert_grad_check<F>(name: &str, inputs: &[Tensor], f: F)
where
    F: Fn(&[Tensor]) -> Tensor,
{
    let result = grad_check(inputs, &f, 1e-3, 2e-2);
    eprintln!(
        "[gradcheck] {name:<14} max|Δ| = {:.2e}",
        result.max_abs_diff
    );
    assert!(
        result.passed,
        "{name}: analytical gradient disagrees with finite differences, max|Δ| = {:.3e}",
        result.max_abs_diff
    );
}
