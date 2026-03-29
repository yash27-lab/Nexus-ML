use nexus_ml::example_model::InferenceModel;
use nexus_ml::io::{save_model, load_model, StatefulModule};

#[test]
fn test_safetensors_serialization() {
    let original_model = InferenceModel::new();
    let state_dict = original_model.state_dict();

    let temp_path = "test_model.safetensors";
    
    // Save
    assert!(save_model(state_dict, temp_path).is_ok());

    // Load
    let loaded_dict = load_model(temp_path).expect("Failed to load safetensors");
    let mut loaded_model = InferenceModel::new(); // Starts with different random weights
    loaded_model.load_state_dict(loaded_dict);

    // Verify
    let original_weight = original_model.fc1.weight.data();
    let loaded_weight = loaded_model.fc1.weight.data();
    
    // Check that weights match exactly after reloading
    assert_eq!(original_weight, loaded_weight);
    
    // Clean up
    std::fs::remove_file(temp_path).unwrap();
}
