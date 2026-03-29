use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use safetensors::tensor::{SafeTensors, TensorView, Dtype};
use crate::Tensor;

pub trait StatefulModule {
    /// Extract parameters mapping parameter names to Tensors
    fn state_dict(&self) -> HashMap<String, Tensor>;
    
    /// Load parameters from a mapping
    fn load_state_dict(&mut self, state_dict: HashMap<String, Tensor>);
}

/// Save a model's state dictionary to a safetensors file
pub fn save_model(state_dict: HashMap<String, Tensor>, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut data_map: HashMap<String, TensorView> = HashMap::new();
    let mut data_storage: Vec<Vec<u8>> = Vec::new(); // Keep data alive

    // Convert our Tensors to safetensors TensorViews
    for (_, tensor) in state_dict.iter() {
        let array = tensor.data();
        
        let mut byte_data = Vec::with_capacity(array.len() * 4);
        for &val in array.iter() {
            byte_data.extend_from_slice(&val.to_ne_bytes());
        }
        data_storage.push(byte_data);
    }
    
    let mut idx = 0;
    for (name, tensor) in state_dict.iter() {
        let array = tensor.data();
        let shape = array.shape().to_vec();
        
        let view = TensorView::new(Dtype::F32, shape, &data_storage[idx])?;
        data_map.insert(name.clone(), view);
        idx += 1;
    }

    let bytes = safetensors::serialize(&data_map, None::<std::collections::HashMap<String, String>>)?;
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
        
        // Convert little-endian bytes back to f32
        let mut f32_data = Vec::with_capacity(data.len() / 4);
        for chunk in data.chunks_exact(4) {
            let bytes: [u8; 4] = chunk.try_into().unwrap();
            f32_data.push(f32::from_ne_bytes(bytes));
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
