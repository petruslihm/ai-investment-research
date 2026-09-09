"""Safe submission entry point: a network-free example unless --app is explicit."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Investment research portfolio - defaults to SYNTHETIC demo")
    parser.add_argument("--demo", action="store_true", help="Synthetic decision demo; never uses provider keys or calls providers")
    parser.add_argument("--app", action="store_true", help="Optional research application; configured providers may be called")
    parser.add_argument("--port", type=int, default=8744)
    parser.add_argument("--data-dir", type=Path, default=Path("data/demo"), help="Isolated demo records")
    args = parser.parse_args()
    if args.app and args.demo:
        parser.error("choose --demo or --app")
    if args.app:
        import sys
        from trading_system.ui.app import main as research_main
        sys.argv = [sys.argv[0]]
        research_main()
        return
    import uvicorn
    from trading_system.demo import create_demo_app
    print(f"DEMO / SYNTHETIC - no market data, LLM calls, training or orders. http://127.0.0.1:{args.port}")
    uvicorn.run(create_demo_app(args.data_dir), host="127.0.0.1", port=args.port)
