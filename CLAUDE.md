We are building an app runtime named BioEngine based on ray. We the way it works is described in bioengine_apps/README.md.

BioEngine app consists of a manifest yaml file and a entry point with python class, defined as `deployments` in the manifest.

We have a deployed version of the BioEngine at https://hypha.aicell.io in workspace `bioimage-io`. When we develop the bioengine apps, we can use this worker to directly test it, to update the exsisting app, you can call something like:
```
python scripts/save_application.py \
    --directory "bioengine_apps/$app" \
    --server-url "https://hypha.aicell.io" \
    --workspace "bioimage-io" \
    --token "$HYPHA_TOKEN"
    ```
In the `.env`, we have stored `HYPHA_TOKEN` for interact with the bioimage-io workspace.

