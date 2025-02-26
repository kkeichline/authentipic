# import
from flask import Flask, render_template, request, redirect, url_for
from authentipic.annotation.manager import AnnotationManager
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

app = Flask(__name__)


class WebAnnotationApp:
    """
    Simple web annotation app interface for manual annotation
    """

    def __init__(self, annotation_manager: AnnotationManager):
        """
        Initialize the web annotation app

        Args:
            annotation_manager: AnnotationManager instance
        """
        self.annotation_manager = annotation_manager

    def serve_app(self, host: str = "localhost", port: int = 8080):
        """
        Serve the annotation app

        Args:
            host: Host to serve on
            port: Port to serve on

        This is a placeholder for the actual web app implementation.
        In a real implementation, this would use Flask, FastAPI, or similar.
        """
        logger.info(f"Starting annotation app on {host}:{port}")
        logger.info("This is a placeholder for the actual web app implementation")

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/annotate", methods=["GET", "POST"])
        def annotate():
            if request.method == "POST":
                # Process annotation
                self.annotation_manager.add_annotation(
                    image_path=request.form["image_path"],
                    label=request.form["label"],
                    annotator_id=request.form["annotator_id"],
                    confidence=int(request.form["confidence"]),
                    annotation_time_sec=float(request.form["annotation_time"]),
                    notes=request.form.get("notes"),
                )
                return redirect(url_for("annotate"))
            else:
                # Get next image to annotate
                next_image = self.annotation_manager.get_next_image_to_annotate(
                    annotator_id=request.args.get("annotator_id", "default")
                )
                return render_template("annotate.html", image_path=next_image)

        @app.route("/stats")
        def stats():
            stats = self.annotation_manager.get_annotation_statistics()
            return render_template("stats.html", stats=stats)

        app.run(host=host, port=port)


def setup_annotation_pipeline(config_path: Optional[str] = None) -> AnnotationManager:
    """
    Set up the annotation pipeline from a config file or defaults

    Args:
        config_path: Path to a JSON configuration file (optional)

    Returns:
        Configured AnnotationManager instance
    """
    config = {}

    if config_path:
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
        except Exception as e:
            logger.warning(f"Error loading config file: {e}. Using defaults.")

    annotation_dir = config.get("annotation_dir", "data/annotations")
    image_dir = config.get("image_dir", "data/raw")
    output_dir = config.get("output_dir", "data/processed/annotations")
    annotation_schema = config.get("annotation_schema", None)

    return AnnotationManager(
        annotation_dir=annotation_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        annotation_schema=annotation_schema,
    )


def main():
    """Main entry point for running the annotation pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="AuthentiPic Annotation Pipeline")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--export", action="store_true", help="Export annotations")
    parser.add_argument(
        "--export-format",
        type=str,
        default="csv",
        choices=["csv", "json", "parquet"],
        help="Export format",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show annotation statistics"
    )
    parser.add_argument(
        "--generate-dataset",
        type=str,
        help="Generate training dataset at specified path",
    )
    parser.add_argument("--web", action="store_true", help="Start web annotation app")
    parser.add_argument(
        "--host", type=str, default="localhost", help="Host for web app"
    )
    parser.add_argument("--port", type=int, default=8080, help="Port for web app")

    args = parser.parse_args()

    # Set up annotation manager
    annotation_manager = setup_annotation_pipeline(args.config)

    # Process commands
    if args.export:
        annotation_manager.export_annotations(format=args.export_format)

    if args.stats:
        stats = annotation_manager.get_annotation_statistics()
        print(json.dumps(stats, indent=2))

    if args.generate_dataset:
        annotation_manager.generate_training_dataset(args.generate_dataset)

    if args.web:
        app = WebAnnotationApp(annotation_manager)
        app.serve_app(host=args.host, port=args.port)

    # If no action specified, print help
    if not any([args.export, args.stats, args.generate_dataset, args.web]):
        parser.print_help()


if __name__ == "__main__":
    main()
