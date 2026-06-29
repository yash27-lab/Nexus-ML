use crate::Tensor;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;

pub trait StatefulModule {
    /// Extract parameters mapping parameter names to Tensors
    fn state_dict(&self) -> HashMap<String, Tensor>;

    /// Load parameters from a mapping
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>);
}

/// Save a model's state dictionary to a safetensors file
pub fn save_model(
    state_dict: HashMap<String, Tensor>,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Materialize names, shapes, and little-endian bytes once, in a single pass,
    // so the view indices stay aligned (no reliance on HashMap iteration order).
    // safetensors is little-endian by spec; `as_standard_layout` guarantees the
    // bytes are C-contiguous row-major even for non-contiguous source arrays.
    let mut names = Vec::with_capacity(state_dict.len());
    let mut shapes = Vec::with_capacity(state_dict.len());
    let mut storage: Vec<Vec<u8>> = Vec::with_capacity(state_dict.len());
    for (name, tensor) in state_dict.iter() {
        let array = tensor.data();
        let standard = array.as_standard_layout();
        let mut bytes = Vec::with_capacity(array.len() * 4);
        for &val in standard.iter() {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        names.push(name.clone());
        shapes.push(array.shape().to_vec());
        storage.push(bytes);
    }

    let mut data_map: HashMap<String, TensorView> = HashMap::new();
    for i in 0..names.len() {
        let view = TensorView::new(Dtype::F32, shapes[i].clone(), &storage[i])?;
        data_map.insert(names[i].clone(), view);
    }

    let bytes =
        safetensors::serialize(&data_map, None::<std::collections::HashMap<String, String>>)?;
    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    Ok(())
}

/// Load a model's state dictionary from a safetensors file
pub fn load_model(path: &str) -> Result<HashMap<String, Tensor>, Box<dyn std::error::Error>> {
    let file_data = std::fs::read(path)?;
    let tensors = SafeTensors::deserialize(&file_data)?;

    let mut state_dict = HashMap::new();

    for (name, view) in tensors.tensors() {
        let data = view.data();

        // Convert little-endian bytes back to f32 (safetensors is LE by spec).
        let mut f32_data = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32_data.push(f32::from_le_bytes(bytes));
        }

        // We assume 2D for simplicity in this PoC, though it could be dynamic
        // Reconstruct ndarray using shape from safetensors
        let shape_vec = view.shape().to_vec();
        let array_d = ndarray::Array::from_shape_vec(shape_vec, f32_data)?.into_dyn();

        let tensor = Tensor::new(array_d, true);
        state_dict.insert(name, tensor);
    }

    Ok(state_dict)
}
