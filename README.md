# BioEngine worker

Manages Ray cluster lifecycle and model deployments on HPC systems or pre-existing Ray environments.

Provides a Hypha service interface for controlling Ray cluster operations, autoscaling, and model deployments through Ray Serve.


## Start BioEngine worker

```bash
bash <(curl -s https://raw.githubusercontent.com/aicell-lab/autoscaler/bioengine-worker/scripts/start_worker.sh) --data_dir <path_to_data> --token <HYPHA_TOKEN>
```

The `HYPHA_TOKEN` can also be stored in the `.env` file in the root directory of the project. The script will automatically load the token from the `.env` file if it exists.