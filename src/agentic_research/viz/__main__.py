"""``python -m agentic_research.viz`` -- launch the dashboard."""

from __future__ import annotations

import argparse
import os
import webbrowser

from agentic_research.viz.server import create_server


def main() -> None:
    p = argparse.ArgumentParser(description="Agentic Research live dashboard")
    p.add_argument("--host", default=os.environ.get("AGENTIC_VIZ_HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("AGENTIC_VIZ_PORT", "8765")))
    p.add_argument("--data-dir", default=os.environ.get("AGENTIC_VIZ_DATA_DIR", ".agentic_viz"),
                   help="Where traces, checkpoints and long-term memory are stored.")
    p.add_argument("--pace", type=float, default=float(os.environ.get("AGENTIC_VIZ_PACE", "0.4")),
                   help="Seconds of delay after visible events, so live runs are legible (0 = full speed).")
    p.add_argument("--open", action="store_true", help="Open the dashboard in your browser.")
    args = p.parse_args()
    server, _ = create_server(args.host, args.port, args.data_dir, pace=args.pace)
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Agentic Research dashboard running at {url}  (Ctrl+C to stop)")
    print(f"Data directory: {args.data_dir}   pace: {args.pace}s")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
