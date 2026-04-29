#!/usr/bin/env python3
"""
BioEngine Datasets - Privacy-preserved scientific data management system.

Serves zarr datasets in-place from a local directory with per-user access
control. Registers an RPC service to the remote central Hypha server for
dataset discovery, and runs a local FastAPI server for efficient zarr chunk
streaming directly to clients.

No data copy is performed — datasets are served from their original location.

Usage:
    python -m bioengine.datasets --data-dir /path/to/datasets --service-token $TOKEN
    python -m bioengine.datasets --data-dir /path/to/datasets --workspace my-workspace
"""

import argparse
import sys

from bioengine.datasets.proxy_server import start_proxy_server


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BioEngine Datasets - Privacy-Preserved Data Streaming Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start datasets service (token from HYPHA_TOKEN env var)
  %(prog)s --data-dir /shared/data

  # Explicit token and workspace
  %(prog)s --data-dir /shared/data --service-token $TOKEN --workspace bioimage-io

  # Custom server port
  %(prog)s --data-dir /shared/data --server-port 8080

For detailed documentation, visit: https://github.com/aicell-lab/bioengine-worker
""",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        metavar="PATH",
        required=True,
        help="Directory containing dataset subdirectories. Each subdirectory must "
        "have a manifest.yaml file defining metadata and access permissions.",
    )
    parser.add_argument(
        "--server-ip",
        type=str,
        metavar="IP_ADDRESS",
        help="IP address for the local file-serving HTTP server. Defaults to the "
        "machine's internal IP.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        metavar="PORT",
        default=9527,
        help="Port for the local HTTP server (default: 9527).",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        metavar="WORKSPACE",
        default="bioimage-io",
        help="Hypha workspace to register the service in (default: bioimage-io).",
    )
    parser.add_argument(
        "--service-token",
        type=str,
        metavar="TOKEN",
        help="Hypha token for registering the service. Falls back to the "
        "HYPHA_TOKEN environment variable if not provided.",
    )
    parser.add_argument(
        "--authentication-server-url",
        type=str,
        metavar="URL",
        default="https://hypha.aicell.io",
        help="URL of the central Hypha server used for token validation "
        "(default: https://hypha.aicell.io).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Path to the log file. Pass 'off' to log to console only. "
        "Defaults to a timestamped file in <data_dir>/../logs/.",
    )

    return parser


if __name__ == "__main__":
    try:
        parser = create_parser()
        args = parser.parse_args()
        kwargs = {k: v for k, v in vars(args).items() if v is not None}
        start_proxy_server(**kwargs)
    except Exception as e:
        print(f"Failed to start BioEngine Datasets proxy server: {e}")
        sys.exit(1)
