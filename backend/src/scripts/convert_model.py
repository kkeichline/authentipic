import argparse
import torch
import os
from pathlib import Path

from authentipic.config import config
from authentipic.models.model_factory import ModelFactory
from authentipic.converter.converter import ModelConverter


def convert_model(
    checkpoint_path: str,
    output_dir: str,
    quantize: bool = True,
    format: str = "tfjs",
    input_shape: tuple = (1, 3, 224, 224)
) -> str:
    """
    Convert a trained PyTorch model to browser/app compatible format.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save the converted model
        quantize: Whether to quantize weights to reduce size
        format: Output format ("tfjs" or "onnx")
        input_shape: Input shape for the model
        
    Returns:
        Path to the converted model
    """
    # Create model
    model = ModelFactory.get_model(config.model)
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # Create converter
    converter = ModelConverter(model, input_shape=input_shape)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert model based on format
    if format.lower() == "tfjs":
        output_path = converter.convert_pipeline(
            output_dir=output_dir,
            quantize=quantize,
            shard_size_bytes=4194304  # 4MB shards for better loading
        )
        print(f"Model converted to TensorFlow.js format at: {output_path}")
    elif format.lower() == "onnx":
        output_path = converter.export_to_onnx(os.path.join(output_dir, "model.onnx"))
        print(f"Model converted to ONNX format at: {output_path}")
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Convert AuthentiPic model for browser/app use")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=config.inference.best_model_path,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./public/models",
        help="Directory to save converted model"
    )
    parser.add_argument(
        "--format", 
        type=str, 
        choices=["tfjs", "onnx"], 
        default="tfjs",
        help="Output format"
    )
    parser.add_argument(
        "--no-quantize", 
        action="store_false", 
        dest="quantize",
        help="Disable quantization (default: enabled)"
    )
    
    args = parser.parse_args()
    
    convert_model(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        quantize=args.quantize,
        format=args.format
    )


if __name__ == "__main__":
    main()