import torch
import os
import subprocess
import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelConverter:
	def __init__(self, model, input_shape=(1, 3, 224, 224)):
		"""
		Initialize the model converter
		
		Args:
			model: PyTorch model to convert
			input_shape: Input shape (batch_size, channels, height, width)
		"""
		self.model = model
		self.input_shape = input_shape
		
	def export_to_onnx(self, output_path="model.onnx"):
		"""
		Export the PyTorch model to ONNX format
		
		Args:
			output_path: Path to save the ONNX model
			
		Returns:
			Path to the saved ONNX model
		"""
		# Create directory if it doesn't exist
		os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
		
		# Create dummy input
		dummy_input = torch.randn(self.input_shape)
		
		# Set model to evaluation mode
		self.model.eval()
		
		# Export to ONNX
		torch.onnx.export(
			self.model,
			dummy_input,
			output_path,
			export_params=True,
			opset_version=13,
			do_constant_folding=True,
			input_names=["input"],
			output_names=["output"],
			dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
		)
		
		logger.info(f"Model exported to ONNX format: {output_path}")
		return output_path
	
	# Alternative implementation using direct APIs
	def convert_to_tensorflow_js(self, output_dir="web_model"):
		"""
		Convert the model to TensorFlow.js format using direct APIs
		
		Args:
			output_dir: Directory to save the TensorFlow.js model
			
		Returns:
			Path to the TensorFlow.js model directory
		"""
		# Export to ONNX
		onnx_path = "temp_model.onnx"
		self.export_to_onnx(onnx_path)
		
		try:
			import tf2onnx
			import tensorflow as tf
			import tensorflowjs as tfjs
			import onnx
			from onnx_tf.backend import prepare
			
			# Load ONNX model
			onnx_model = onnx.load(onnx_path)
			# Prepare tf representation
			tf_rep = prepare(onnx_model)
			
			# Create a TensorFlow SavedModel
			tf_output_dir = "temp_tf_model"
			os.makedirs(tf_output_dir, exist_ok=True)
			tf_rep.export_graph(tf_output_dir)
			
			# Convert TensorFlow SavedModel to TensorFlow.js
			os.makedirs(output_dir, exist_ok=True)
			tfjs.converters.save_keras_model(
				tf.saved_model.load(tf_output_dir),
				output_dir
			)
			
			# Clean up
			if os.path.exists(onnx_path):
				os.remove(onnx_path)
			if os.path.exists(tf_output_dir):
				shutil.rmtree(tf_output_dir)
				
			return output_dir
		
		except Exception as e:
			logger.error(f"Error in TensorFlow.js conversion: {e}")
			if os.path.exists(onnx_path):
				logger.info(f"Temporary ONNX file remains at: {onnx_path}")
			raise RuntimeError(f"Failed to convert model to TensorFlow.js: {str(e)}")
	
	def convert_to_tflite(self, output_path="model.tflite", optimize=True):
		"""
		Convert the model to TensorFlow Lite format for mobile apps
		
		Args:
			output_path: Path to save the TFLite model
			optimize: Whether to optimize the model for mobile
			
		Returns:
			Path to the TensorFlow Lite model
		"""
		# First export to ONNX
		onnx_path = "temp_model.onnx"
		self.export_to_onnx(onnx_path)
		
		try:
			# Convert ONNX to TensorFlow frozen model
			tf_output_dir = "temp_tf_model"
			os.makedirs(tf_output_dir, exist_ok=True)
			
			# Use tf2onnx to convert ONNX to TensorFlow with correct params
			cmd = [
				"python", "-m", "tf2onnx.convert",
				"--input", onnx_path,
				"--output", os.path.join(tf_output_dir, "model.pb"),
				"--target", "tf",
				"--output_frozen_graph"
			]
			
			logger.info(f"Running command: {' '.join(cmd)}")
			subprocess.run(cmd, check=True)
			
			# Convert TF frozen model to TFLite
			# We'll use the tflite_convert command line tool which is more reliable
			tflite_convert_cmd = [
				"tflite_convert",
				"--graph_def_file", os.path.join(tf_output_dir, "model.pb"),
				"--output_file", output_path,
				"--input_arrays", "input",
				"--output_arrays", "output"
			]
			
			if optimize:
				tflite_convert_cmd.append("--post_training_quantize")
				
			logger.info(f"Running TFLite conversion: {' '.join(tflite_convert_cmd)}")
			subprocess.run(tflite_convert_cmd, check=True)
			
			logger.info(f"Model converted to TensorFlow Lite: {output_path}")
			
			# Clean up temporary files
			if os.path.exists(onnx_path):
				os.remove(onnx_path)
			if os.path.exists(tf_output_dir):
				shutil.rmtree(tf_output_dir)
				
			return output_path
			
		except Exception as e:
			logger.error(f"Error in TFLite conversion: {e}")
			if os.path.exists(onnx_path):
				logger.info(f"Temporary ONNX file remains at: {onnx_path}")
			raise RuntimeError(f"Failed to convert model to TFLite: {str(e)}")
			
	def convert_to_coreml(self, output_path="model.mlmodel"):
		"""
		Convert the model to CoreML format for iOS apps
		
		Args:
			output_path: Path to save the CoreML model
			
		Returns:
			Path to the CoreML model
		"""
		# First export to ONNX
		onnx_path = "temp_model.onnx"
		self.export_to_onnx(onnx_path)
		
		try:
			# Check if coremltools is available
			import coremltools as ct
			
			# Load ONNX model
			onnx_model = ct.converters.onnx.load(onnx_path)
			
			# Convert to CoreML
			mlmodel = ct.converters.onnx.convert(
				model=onnx_model,
				minimum_ios_deployment_target="13"
			)
			
			# Save the model
			os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
			mlmodel.save(output_path)
			
			logger.info(f"Model converted to CoreML: {output_path}")
			
			# Clean up temporary files
			if os.path.exists(onnx_path):
				os.remove(onnx_path)
				
			return output_path
			
		except ImportError:
			logger.error("CoreMLTools not available. Install with: pip install coremltools")
			raise
		except Exception as e:
			logger.error(f"Error in CoreML conversion: {e}")
			raise
			
	def convert_all(self, output_dir="converted_models"):
		"""
		Convert the model to all supported formats
		
		Args:
			output_dir: Base directory to save converted models
			
		Returns:
			Dictionary of paths to converted models
		"""
		os.makedirs(output_dir, exist_ok=True)
		
		results = {}
		
		# Export ONNX (useful by itself)
		onnx_path = os.path.join(output_dir, "model.onnx")
		results["onnx"] = self.export_to_onnx(onnx_path)
		
		# Convert to TensorFlow.js
		try:
			tfjs_dir = os.path.join(output_dir, "web_model")
			results["tensorflow_js"] = self.convert_to_tensorflow_js(tfjs_dir)
		except Exception as e:
			logger.error(f"Failed to convert to TensorFlow.js: {e}")
		
		# Convert to TFLite
		try:
			tflite_path = os.path.join(output_dir, "model.tflite")
			results["tflite"] = self.convert_to_tflite(tflite_path)
		except Exception as e:
			logger.error(f"Failed to convert to TensorFlow Lite: {e}")
		
		# Convert to CoreML
		try:
			coreml_path = os.path.join(output_dir, "model.mlmodel")
			results["coreml"] = self.convert_to_coreml(coreml_path)
		except Exception as e:
			logger.error(f"Failed to convert to CoreML: {e}")
		
		logger.info(f"Model conversion complete. Results: {results}")
		return results