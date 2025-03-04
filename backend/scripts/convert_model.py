import argparse
import os
import torch
from pathlib import Path
import logging

from authentipic.config import config
from authentipic.models.model_factory import ModelFactory
from authentipic.converter.converter import ModelConverter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("convert_model")

def convert_model(
    checkpoint_path,
    output_dir,
    format_type="all",
    model_type=config.model.architecture, 
    num_classes=config.model.num_classes,
    input_shape=(1, 3, 224, 224),
    quantize=True,
    shard_size_bytes=4194304  # 4MB shards
):
    """
    Convert a PyTorch model to specified format(s).
    
    Args:
        checkpoint_path: Path to PyTorch model checkpoint
        output_dir: Directory to save converted model
        format_type: Type of format to convert to (tfjs, onnx, tflite, coreml, all)
        model_type: Model architecture type
        num_classes: Number of output classes
        input_shape: Input tensor shape
        quantize: Whether to quantize weights (for tfjs and tflite)
        shard_size_bytes: Size of weight shards for tfjs
    
    Returns:
        Path to the converted model
    """
    logger.info(f"Converting model from {checkpoint_path} to {format_type} format")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model
    model_config = config.model
    model_config.architecture = model_type
    model_config.num_classes = num_classes
    model = ModelFactory.get_model(model_config)
    
    # Load weights if checkpoint exists
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            # Check if model state is in "model_state_dict" key or directly
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded model state from checkpoint: {checkpoint_path}")
            else:
                model.load_state_dict(checkpoint)
                logger.info(f"Loaded raw state dict from checkpoint: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.warning(f"Proceeding with untrained model")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}. Using untrained model.")
    
    # Set model to evaluation mode
    model.eval()
    
    # Create converter
    converter = ModelConverter(model, input_shape=input_shape)
    
    # Convert model based on requested format
    if format_type == "all":
        logger.info("Converting to all supported formats")
        return converter.convert_all(output_dir=str(output_dir))
    
    elif format_type == "onnx":
        onnx_path = output_dir / "model.onnx"
        logger.info(f"Converting to ONNX format: {onnx_path}")
        return converter.export_to_onnx(output_path=str(onnx_path))
    
    elif format_type == "tfjs":
        tfjs_dir = output_dir / "web_model"
        logger.info(f"Converting to TensorFlow.js format: {tfjs_dir}")
        return converter.convert_to_tensorflow_js(output_dir=str(tfjs_dir))
    
    elif format_type == "tflite":
        tflite_path = output_dir / "model.tflite"
        logger.info(f"Converting to TensorFlow Lite format: {tflite_path}")
        return converter.convert_to_tflite(output_path=str(tflite_path), optimize=quantize)
    
    elif format_type == "coreml":
        coreml_path = output_dir / "model.mlmodel"
        logger.info(f"Converting to CoreML format: {coreml_path}")
        return converter.convert_to_coreml(output_path=str(coreml_path))
    
    else:
        logger.error(f"Unsupported format type: {format_type}")
        raise ValueError(f"Unsupported format type: {format_type}")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to various formats")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default=config.inference.best_model_path,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="./models/converted",
        help="Output directory for converted model"
    )
    parser.add_argument(
        "--format", 
        type=str,
        default="all",
        choices=["tfjs", "onnx", "tflite", "coreml", "all"],
        help="Format to convert to"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        default="resnet50", 
        help="Model architecture type"
    )
    parser.add_argument(
        "--num-classes", 
        type=int, 
        default=2,
        help="Number of output classes"
    )
    parser.add_argument(
        "--no-quantize", 
        action="store_false",
        dest="quantize", 
        help="Disable quantization (default: enabled)"
    )
    parser.add_argument(
        "--shard-size", 
        type=int,
        default=4194304,  # 4MB
        help="Size of weight shards in bytes for TensorFlow.js (default: 4MB)"
    )
    
    args = parser.parse_args()
    
    try:
        result = convert_model(
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
            format_type=args.format,
            model_type=args.model_type,
            num_classes=args.num_classes,
            quantize=args.quantize,
            shard_size_bytes=args.shard_size
        )
        logger.info(f"Conversion successful. Result: {result}")
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit(1)

if __name__ == "__main__":
    main()