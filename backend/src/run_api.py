import uvicorn
import argparse


def main():
    parser = argparse.ArgumentParser(description="Run AuthentiPic API server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the API server on"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to run the API server on"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )

    args = parser.parse_args()

    print(f"Starting AuthentiPic API server on {args.host}:{args.port}")
    uvicorn.run(
        "authentipic.api:app", host=args.host, port=args.port, reload=args.reload
    )


if __name__ == "__main__":
    main()
