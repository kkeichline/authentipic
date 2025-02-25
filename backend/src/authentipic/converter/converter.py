import torch
import tensorflowjs as tfjs
import os
from pathlib import Path

class ModelConverter:
    def __init__(self, model, input_shape=(1, 3, 224, 224)):
        """
        Initialize the converter with a PyTorch model
        
        Args:
            model: PyTorch model to convert
            input_shape: Expected input shape (batch_size, channels, height, width)
        """
        self.model = model
        self.input_shape = input_shape
        
    def export_to_onnx(self, output_path="model.onnx"):
        """
        Export PyTorch model to ONNX format
        
        Args:
            output_path: Path to save the ONNX model
        """
        # Create dummy input for tracing
        dummy_input = torch.randn(self.input_shape)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=12  # Use recent opset version for better compatibility
        )
        
        return output_path
    
    def convert_to_tfjs(self, 
                       onnx_path="model.onnx", 
                       output_dir="./public/models",
                       quantize=False,
                       shard_size_bytes=None):
        """
        Convert ONNX model to TensorFlow.js format
        
        Args:
            onnx_path: Path to the ONNX model
            output_dir: Directory to save the TFJS model
            quantize: Whether to quantize the weights to reduce model size
            shard_size_bytes: Size of weight shards in bytes (optional)
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Prepare conversion arguments
        conversion_args = {
            'input_format': 'onnx',
            'output_format': 'tfjs_layers_model',
            'signature_name': 'serving_default',
            'quantization_dtype': 'uint8' if quantize else None,
            'weight_shard_size_bytes': shard_size_bytes
        }
        
        # Convert to TFJS format
        tfjs.converters.convert.convert_tf_saved_model(
            onnx_path,
            output_dir,
            **{k: v for k, v in conversion_args.items() if v is not None}
        )
        
        return output_dir
    
    def convert_pipeline(self, 
                        output_dir="./public/models",
                        quantize=False,
                        shard_size_bytes=None,
                        cleanup_onnx=True):
        """
        Run the full conversion pipeline from PyTorch to TFJS
        
        Args:
            output_dir: Directory to save the TFJS model
            quantize: Whether to quantize the weights
            shard_size_bytes: Size of weight shards in bytes
            cleanup_onnx: Whether to delete the intermediate ONNX file
        """
        try:
            # Export to ONNX
            onnx_path = self.export_to_onnx()
            
            # Convert to TFJS
            tfjs_dir = self.convert_to_tfjs(
                onnx_path=onnx_path,
                output_dir=output_dir,
                quantize=quantize,
                shard_size_bytes=shard_size_bytes
            )
            
            # Cleanup ONNX file if requested
            if cleanup_onnx and os.path.exists(onnx_path):
                os.remove(onnx_path)
                
            return tfjs_dir
            
        except Exception as e:
            print(f"Error during conversion: {str(e)}")
            raise

# Example usage:
"""
# Initialize your PyTorch model
model = YourPyTorchModel()
model.load_state_dict(torch.load('model_weights.pth'))

# Create converter
converter = ModelConverter(model)

# Run conversion pipeline
converter.convert_pipeline(
    output_dir='./frontend/public/models',
    quantize=True,
    shard_size_bytes=4194304  # 4MB shards
)
"""