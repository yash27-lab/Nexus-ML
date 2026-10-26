use ndarray::array;
use nexus_ml::io::{load_model, save_model};
use nexus_ml::Tensor;
use std::collections::HashMap;

#[test]
fn reject_integer() {
    use safetensors::tensor::{Dtype, TensorView};
    let f = Fixture::new();
    let bytes = 42_i32.to_le_bytes();
    let view = TensorView::new(Dtype::I32, vec![1], &bytes).unwrap();
    let data = safetensors::serialize([("weight", view)], None).unwrap();
    std::fs::write(&f.0, data).unwrap();
    let err = load_model(f.path()).err().expect("integer dtype must fail");
    assert!(err.to_string().contains("expected F32"));
}

struct Fixture(std::path::PathBuf);
impl Fixture {
    fn new() -> Self {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "nexus_checkpoint_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        )))
    }
    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}
