import argparse
import json
from pathlib import Path
from .experiment import ExperimentConfig, run_experiment, save_result


def main():
    parser = argparse.ArgumentParser(description="Interval local research runtime")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path)
    run.add_argument("--output", type=Path, default=Path("runtime/latest"))
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8765)
    train = sub.add_parser("train-planner")
    train.add_argument("--train-data", type=Path)
    train.add_argument("--test-data", type=Path)
    train.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    if args.command == "run":
        config = ExperimentConfig.model_validate_json(args.config.read_text()) if args.config else ExperimentConfig()
        result, fitted = run_experiment(config, lambda stage, message, detail: print(f"{stage}: {message}", flush=True))
        save_result(result, args.output, fitted)
        print(json.dumps({"output": str(args.output.resolve()), "seconds": result["seconds"]}))
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("interval.server:app", host="127.0.0.1", port=args.port)
    else:
        from .training import train_planner
        receipt = train_planner(steps=args.steps, train_path=args.train_data, test_path=args.test_data)
        print(json.dumps({key: receipt[key] for key in ("model", "steps", "device", "seconds", "adapter_sha256")}))


if __name__ == "__main__":
    main()
