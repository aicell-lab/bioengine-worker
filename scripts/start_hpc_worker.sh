# !/bin/bash

# Extract version from pyproject.toml (single source of truth)
get_version_from_pyproject() {
    local pyproject_path="$1/pyproject.toml"
    if [[ -f "$pyproject_path" ]]; then
        grep -E '^version\s*=' "$pyproject_path" | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1
    fi
}

# Get version from GitHub API for remote execution
get_version_from_github() {
    local github_url="https://raw.githubusercontent.com/aicell-lab/bioengine-worker/main/pyproject.toml"
    if command -v curl >/dev/null 2>&1; then
        curl -s "$github_url" | grep -E '^version\s*=' | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$github_url" | grep -E '^version\s*=' | sed -E 's/version\s*=\s*"(.*)"/\1/' | head -1
    fi
}

# Determine the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Get version from pyproject.toml (local) or GitHub (remote)
VERSION=$(get_version_from_pyproject "$PROJECT_ROOT")

if [[ -z "$VERSION" ]]; then
    # If no local pyproject.toml, try to fetch from GitHub
    echo "Local pyproject.toml not found, fetching version from GitHub..."
    VERSION=$(get_version_from_github)
    
    if [[ -z "$VERSION" ]]; then
        echo "❌ Error: Could not determine version from local pyproject.toml or GitHub."
        echo "   This script requires either:"
        echo "   1. Running from a cloned bioengine-worker repository, or"
        echo "   2. Internet access to fetch version from GitHub"
        exit 1
    else
        echo "✅ Found version $VERSION from GitHub"
    fi
else
    echo "✅ Found version $VERSION from local pyproject.toml"
fi

DEFAULT_IMAGE="ghcr.io/aicell-lab/bioengine-worker:$VERSION"
WORKING_DIR=$(pwd)

# Save all arguments
BIOENGINE_WORKER_ARGS=("$@")

# === Determine container command ===

if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "Neither Apptainer nor Singularity could be found. Please install one of them first."
    exit 1
fi

# === Define functions for argument handling ===

# Function to get argument value
get_arg_value() {
    local tag="$1"
    local default="$2"

    # Check if the tag is present in the arguments
    local value="$default"
    for ((i=0; i<${#BIOENGINE_WORKER_ARGS[@]}; i++)); do
        if [[ "${BIOENGINE_WORKER_ARGS[i]}" == "$tag" ]] && [ $((i+1)) -lt ${#BIOENGINE_WORKER_ARGS[@]} ]; then
            value="${BIOENGINE_WORKER_ARGS[i+1]}"
            break
        elif [[ "${BIOENGINE_WORKER_ARGS[i]}" == "$tag="* ]]; then
            value="${BIOENGINE_WORKER_ARGS[i]#*=}"
            break
        fi
    done

    echo "$value"
}

# Function to set/update an argument value
set_arg_value() {
    local tag="$1"
    local value="$2"
    local found=false
    
    # Look for existing argument
    for ((i=0; i<${#BIOENGINE_WORKER_ARGS[@]}; i++)); do
        if [[ "${BIOENGINE_WORKER_ARGS[i]}" == "$tag" ]] && [ $((i+1)) -lt ${#BIOENGINE_WORKER_ARGS[@]} ]; then
            # Update space-separated format
            BIOENGINE_WORKER_ARGS[i+1]="$value"
            found=true
            break
        elif [[ "${BIOENGINE_WORKER_ARGS[i]}" == "$tag="* ]]; then
            # Update equals format
            BIOENGINE_WORKER_ARGS[i]="$tag=$value"
            found=true
            break
        fi
    done
    
    # Add new argument if not found
    if [ "$found" = false ]; then
        BIOENGINE_WORKER_ARGS+=("$tag" "$value")
    fi
}

# Function to define bind mounts
BIND_OPTS=()
add_bind() {
    if [[ ! -e "$1" ]]; then
        echo "Warning: $1 does not exist."
        exit 1
    fi
    if [[ $# -eq 1 ]]; then
        BIND_OPTS+=("--bind=$1")
    elif [[ $# -eq 2 ]]; then
        BIND_OPTS+=("--bind=$1:$2")
    elif [[ $# -eq 3 ]]; then
        BIND_OPTS+=("--bind=$1:$2:$3")
    else
        echo "Error: Invalid number of arguments for add_bind function."
        exit 1
    fi
}

# Function to define environment variables
ENV_VARS=()
add_env() {
    if [[ -n "$2" ]]; then
        ENV_VARS+=("--env=$1=$2")
    fi
}

# === Ensure mode is set to "slurm" ===

# Check if the mode is set to something else than "slurm"
MODE=$(get_arg_value "--mode" "slurm")
if [[ "$MODE" != "slurm" ]]; then
    echo "Error: Invalid mode '$MODE'. For modes other than 'slurm', please run the 'bioengine-worker' container directly. Check out the configuration wizard at https://bioimage.io/#/bioengine."
    exit 1
fi

# === Load BioEngine image ===

# Get the container cache directory
CACHE_DIR=$(get_arg_value "--cache_dir" "$WORKING_DIR/.bioengine")
CACHE_DIR=$(realpath $CACHE_DIR)

IMAGE_CACHEDIR=$(get_arg_value "--apptainer_cachedir")
if [[ -z "$IMAGE_CACHEDIR" ]]; then
    IMAGE_CACHEDIR=$(get_arg_value "--singularity_cachedir" "$CACHE_DIR/images")
fi
IMAGE_CACHEDIR=$(realpath $IMAGE_CACHEDIR)
mkdir -p $IMAGE_CACHEDIR
SINGULARITY_CACHEDIR=$IMAGE_CACHEDIR
APPTAINER_CACHEDIR=$IMAGE_CACHEDIR

# Get the path to the image 
IMAGE="$(get_arg_value "--image" $DEFAULT_IMAGE)"

# Get the image name and version
if [[ "$IMAGE" == *.sif ]]; then
    # Check if local Singularity image file exists
    IMAGE=$(realpath $IMAGE)
    if [ ! -f "$IMAGE" ]; then
        echo "Error: Image file $IMAGE not found."
        exit 1
    fi
elif [[ "$IMAGE" != docker://* ]]; then
    # Add docker:// prefix if not present
    IMAGE="docker://${IMAGE}"
fi
set_arg_value "--image" $IMAGE

# === Set up bind mounts ===

# Add SLURM-specific bindings
if [[ "$MODE" == "slurm" ]]; then
    # Binaries
    add_bind $(which sinfo)
    add_bind $(which sbatch)
    add_bind $(which squeue)
    add_bind $(which scancel)

    # Configuration files
    add_bind "/etc/hosts"
    add_bind "/etc/localtime"
    add_bind "/etc/passwd"
    add_bind "/etc/group"
    add_bind "/etc/slurm"
    add_bind "/etc/munge"

    # SLURM and Munge libraries
    add_bind "/usr/lib64/slurm"
    for lib in /usr/lib64/libmunge.so*; do
        add_bind "$lib"
    done

    # Munge sockets
    add_bind "/var/run/munge"
    # Munge key
    add_bind "/var/lib/munge"
    # Munge logs
    add_bind "/var/log/munge"
fi


# Add BioEngine worker bindings

# CACHE_DIR is needed by the BioEngine worker -> container path
mkdir -p $CACHE_DIR
add_bind $CACHE_DIR "/tmp/bioengine"
set_arg_value "--cache_dir" "/tmp/bioengine"

# Pass the real cache_dir to the SLURM worker node via --worker_cache_dir
WORKER_CACHE_DIR=$(get_arg_value "--worker_cache_dir" $CACHE_DIR)
WORKER_CACHE_DIR=$(realpath $WORKER_CACHE_DIR)
set_arg_value "--worker_cache_dir" $WORKER_CACHE_DIR

# DATA_DIR is needed on by the DatasetsManager -> container path
DATA_DIR=$(get_arg_value "--data_dir")
if [[ -n "$DATA_DIR" ]]; then
    DATA_DIR=$(realpath $DATA_DIR)
    add_bind $DATA_DIR "/data" "ro"  # Read-only
    set_arg_value "--data_dir" /data
fi

# Pass the real data_dir to the SLURM worker node via --worker_data_dir
WORKER_DATA_DIR=$(get_arg_value "--worker_data_dir" $DATA_DIR)
if [[ -n "$WORKER_DATA_DIR" ]]; then
    WORKER_DATA_DIR=$(realpath $WORKER_DATA_DIR)
    set_arg_value "--worker_data_dir" $WORKER_DATA_DIR
fi

# Check if the flag `--debug` is set
DEBUG_MODE=$(get_arg_value "--debug" "false")
if [[ ! "$DEBUG_MODE" == "false" ]]; then
    echo "Debug mode is enabled. Binding current working directory ($WORKING_DIR) to /app in the container."
    echo ""

    # Add debug bindings
    add_bind $WORKING_DIR "/app"

    echo "Starting BioEngine worker with the following arguments:"
    for arg in "${BIOENGINE_WORKER_ARGS[@]}"; do
        echo "  $arg"
    done
    echo ""

    echo "Starting BioEngine worker with the following environment variables:"
    for env in "${ENV_VARS[@]}"; do
        echo "  $env"
    done
    echo ""

    echo "Starting BioEngine worker with the following bind mounts:"
    for bind in "${BIND_OPTS[@]}"; do
        echo "  $bind"
    done
    echo ""
fi

# === Set up environment variables ===

# Export environment variables from .env file if it exists
if [ -f "$WORKING_DIR/.env" ]; then
    set -a
    source $WORKING_DIR/.env
    set +a
fi

# Add environment variables
add_env "USER" "$USER"

# Add Hypha token if available
if [ -n "$HYPHA_TOKEN" ]; then
    add_env "HYPHA_TOKEN" "$HYPHA_TOKEN"
fi

# === Set up cleanup ===

cleanup() {
    # TODO: try to prevent SIGINT and SIGTERM from being sent to the container, then call service.cleanup() using http API
    echo "Making sure the Ray head node is stopped..."
    $CONTAINER_CMD exec "$IMAGE_PATH" ray stop --force

    # If running in SLURM mode, cancel any remaining SLURM jobs
    if [[ "$MODE" == "slurm" ]]; then
        echo "Cleaning up any remaining Ray worker jobs..."
        WORKER_JOB_IDS=$(squeue -u $USER -n "ray_worker" -h -o "%i")
        if [[ -n "$WORKER_JOB_IDS" ]]; then
            while read -r jobid; do
                scancel $jobid
            done <<< "$WORKER_JOB_IDS"
            echo "All Ray worker jobs have been successfully canceled."
        else
            echo "No Ray worker jobs found to cancel."
        fi
    fi
}

# Set trap to ensure cleanup runs on script exit (normal or abnormal)
trap cleanup EXIT

# === Run the BioEngine worker ===

# Run with clean environment
$CONTAINER_CMD exec \
    --cleanenv \
    --pwd /app \
    "${ENV_VARS[@]}" \
    "${BIND_OPTS[@]}" \
    "$IMAGE" \
    python -m bioengine_worker "${BIOENGINE_WORKER_ARGS[@]}"