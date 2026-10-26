use ndarray::array;
use nexus_ml::io::{load_model, save_model};
use nexus_ml::Tensor;
use std::collections::HashMap;

#[test]
fn malformed_checkpoint() {
    let f = Fixture::new();
    for bytes in [
        b"".as_slice(),
        b"not a safetensors checkpoint".as_slice(),
        &[255; 8],
    ] {
        std::fs::write(&f.0, bytes).unwrap();
        assert!(load_model(f.path()).is_err());
    }
}

#[test]
fn strided_roundtrip() {
    let f = Fixture::new();
    let data = array![[1.0, 2.0], [3.0, 4.0]].reversed_axes().into_dyn();
    let mut state = HashMap::new();
    state.insert("weight".into(), Tensor::new(data.clone(), true));
    save_model(state, f.path()).unwrap();
    let loaded = load_model(f.path()).unwrap();
    assert_eq!(loaded["weight"].data(), data);
}

#[test]
fn scalar_roundtrip() {
    let f = Fixture::new();
    let t = Tensor::new(ndarray::arr0(3.5).into_dyn(), true);
    let mut state = HashMap::new();
    state.insert("scalar".into(), t);
    save_model(state, f.path()).unwrap();
    let loaded = load_model(f.path()).unwrap();
    assert_eq!(loaded["scalar"].data().ndim(), 0);
    assert_eq!(loaded["scalar"].data()[[]], 3.5);
}

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
