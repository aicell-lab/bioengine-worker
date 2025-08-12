# %% Imports
from pathlib import Path

import anndata


# %% Define Functions
def save_h5ad_to_zarr(fpath):
    fpath = Path(fpath).resolve()
    print(f"Saving {fpath} to zarr...", end="")

    # Load the anndata files
    adata = anndata.read_h5ad(fpath)

    print(
        f"\rSaving {fpath} with n_samples {adata.n_obs} and n_vars {adata.n_vars} to zarr..."
    )

    # Save anndata as zarr
    adata.write_zarr(fpath.with_suffix(".zarr"))


def list_anndata_files(data_dir):
    # List all h5ad files in the data directory and subdirectories
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise ValueError(f"{data_dir} is not a directory")

    h5ad_files = list(data_dir.rglob("*.h5ad"))

    return h5ad_files


# %% Save each anndata dataset to zarr

data_dir = Path(__file__).parent.parent / "data"

for fpath in list_anndata_files(data_dir):
    # Save the anndata object to zarr
    save_h5ad_to_zarr(fpath)

# %%
