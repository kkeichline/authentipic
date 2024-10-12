from authentipic.data import make_dataset, preprocess
from authentipic.features import build_features
from authentipic.models import train_model, predict_model

def run_training_pipeline():
    # Orchestrate the entire training process
    pass

def run_inference_pipeline(image_path: str):
    # Orchestrate the inference process for a single image
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict"])
    parser.add_argument("--image_path", help="Path to image for prediction")
    args = parser.parse_args()

    if args.mode == "train":
        run_training_pipeline()
    elif args.mode == "predict":
        run_inference_pipeline(args.image_path)