"""
BioEngine CLI — main entry point.

Usage:
  bioengine runner search --keywords nuclei segmentation
  bioengine runner infer <model-id> --input image.tif --output result.npy
  bioengine apps deploy ./my-app/
  bioengine apps status

Environment variables:
  BIOENGINE_SERVER_URL          Hypha server URL (default: https://hypha.aicell.io)
  BIOENGINE_WORKER_SERVICE_ID   BioEngine worker service ID (for apps commands)
  HYPHA_TOKEN / BIOENGINE_TOKEN Authentication token (for apps commands)
"""
import click

from bioengine_cli import __version__
from bioengine_cli.runner import runner_group
from bioengine_cli.apps import apps_group


@click.group()
@click.version_option(version=__version__, prog_name="bioengine")
def main():
    """
    BioEngine — run AI models for microscopy image analysis.

    BioEngine is a distributed platform for finding, running, and deploying
    deep learning models for bioimaging. Models run on remote GPU workers —
    no local GPU needed.

    \b
    Getting started:
      bioengine runner search --keywords nuclei segmentation
      bioengine runner info <model-id>
      bioengine runner infer <model-id> --input image.tif --output result.npy

    \b
    Deploying custom apps:
      bioengine apps deploy ./my-app/   (requires worker access + token)
      bioengine apps list
      bioengine apps status <app-id>

    \b
    Environment variables:
      BIOENGINE_SERVER_URL          Server URL (default: https://hypha.aicell.io)
      BIOENGINE_WORKER_SERVICE_ID   Worker service ID (for apps commands)
      HYPHA_TOKEN                   Auth token (for apps commands)

    Run `bioengine runner --help` or `bioengine apps --help` for details.
    """


main.add_command(runner_group)
main.add_command(apps_group)


if __name__ == "__main__":
    main()
