"""Check the status of a Cellpose training session and display metrics."""

from __future__ import annotations

import argparse
import asyncio
import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv

from hypha_rpc import connect_to_server, login

if TYPE_CHECKING:
    from hypha_rpc.rpc import RemoteService

# load .env
load_dotenv()


async def check_status(session_id: str) -> None:
    """Check and display the status of a training session."""
    server_url = os.environ.get("HYPHA_SERVER_URL", "https://hypha.aicell.io")
    token = os.environ.get("HYPHA_TOKEN") or await login({"server_url": server_url})

    async with connect_to_server({"server_url": server_url, "token": token}) as server:  # type: ignore[generalTypeIssues]
        cellpose_service = await server.get_service("bioimage-io/cellpose-finetuning")
        status = await cellpose_service.get_training_status(session_id)

        print(f"\nTraining Session: {session_id}")
        print(f"Status: {status['status_type']}")
        print(f"Message: {status['message']}")

        # Display training metrics if available
        if "train_losses" in status and status["train_losses"]:
            train_losses = status["train_losses"]
            non_zero_train = [loss for loss in train_losses if loss > 0]

            if non_zero_train:
                print("\nTraining Metrics:")
                print(f"  Total epochs: {len(non_zero_train)}")
                print(f"  Initial train loss: {non_zero_train[0]:.4f}")
                print(f"  Current train loss: {non_zero_train[-1]:.4f}")
                print(f"  Improvement: {(non_zero_train[0] - non_zero_train[-1]):.4f}")

                if "test_losses" in status and status["test_losses"]:
                    test_losses = status["test_losses"]
                    non_zero_test = [loss for loss in test_losses if loss > 0]
                    if non_zero_test:
                        print(f"\n  Test evaluations: {len(non_zero_test)}")
                        print(f"  Latest test loss: {non_zero_test[-1]:.4f}")

                # Print loss history
                print("\n  Loss History:")
                for i, loss in enumerate(non_zero_train):
                    msg = f"    Epoch {i+1}: train={loss:.4f}"
                    if i < len(test_losses) and test_losses[i] > 0:
                        msg += f", test={test_losses[i]:.4f}"
                    print(msg)
        else:
            print("\nNo training metrics available yet.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check the status of a Cellpose training session.",
    )
    parser.add_argument(
        "session_id",
        type=str,
        help="Training session ID to check",
    )
    args = parser.parse_args()

    asyncio.run(check_status(args.session_id))
