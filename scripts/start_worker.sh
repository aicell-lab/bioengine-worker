# !/bin/bash

VERSION=0.1.16
DEFAULT_IMAGE="ghcr.io/aicell-lab/bioengine-worker:$VERSION"
WORKING_DIR=$(pwd)

# Save all arguments
BIOENGINE_WORKER_ARGS=("$@")

# Check if Apptainer or Singularity is installed
if command -v apptainer &> /dev/null; then
    CONTAINER_CMD="apptainer"
elif command -v singularity &> /dev/null; then
    CONTAINER_CMD="singularity"
else
    echo "Neither Apptainer nor Singularity could be found. Please install one of them first."
    exit 1
fi

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

# Get the path to the image
IMAGE="$(get_arg_value "--image" $DEFAULT_IMAGE)"

# Get the image name and version
if [[ "$IMAGE" == *.sif ]]; then
    # Local Singularity image
    IMAGE_PATH=$(realpath $IMAGE)
    DOCKER_IMAGE=
else
    # Remote Docker image
    FILE=${IMAGE##*/}
    NAME=${FILE%%:*}
    VERSION=${FILE##*:}

    IMAGE_PATH="$WORKING_DIR/apptainer_images/${NAME}_${VERSION}.sif"
    DOCKER_IMAGE=$IMAGE
fi

# Check if the BioEngine worker image is available
if [ ! -f "$IMAGE_PATH" ]; then
    if [[ -z "$DOCKER_IMAGE" ]]; then
        echo "Error: Image file $IMAGE_PATH not found."
        exit 1
    fi
    # Ask user before pulling
    read -p "Image file $IMAGE_PATH not found. Do you want to pull it from Docker? (yes/no): " response
    if [[ "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Pulling image $DOCKER_IMAGE..."
        IMAGE_DIR=$(dirname "$IMAGE_PATH")
        mkdir -p $IMAGE_DIR
        $CONTAINER_CMD pull $IMAGE_PATH docker://$DOCKER_IMAGE
        if [[ $? -eq 0 ]]; then
            echo "Successfully pulled $DOCKER_IMAGE to $IMAGE_PATH"
        else
            echo "Error pulling image $DOCKER_IMAGE"
            exit 1
        fi
    else
        echo "Image download cancelled"
        exit 1
    fi
fi

set_arg_value "--image" $IMAGE_PATH

# Define environment variables
ENV_VARS=()
add_env() {
    if [[ -n "$2" ]]; then
        ENV_VARS+=("--env=$1=$2")
    fi
}

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

# Define bind options
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

# Add SLURM-specific bindings if SLURM is used
MODE=$(get_arg_value "--mode" "slurm")
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

# LOG_DIR is needed by the BioEngine logger -> container path
LOG_DIR=$(get_arg_value "--log_dir" "$WORKING_DIR/logs")
LOG_DIR=$(realpath $LOG_DIR)
mkdir -p $LOG_DIR
add_bind $LOG_DIR "/app/logs"
set_arg_value "--log_dir" "/app/logs"

# DATA_DIR is needed on by the DatasetManager -> container path
DATA_DIR=$(get_arg_value "--data_dir" "")
if [[ -n "$DATA_DIR" ]]; then
    DATA_DIR=$(realpath $DATA_DIR)
    add_bind $DATA_DIR "/data" "ro"  # Read-only
    set_arg_value "--data_dir" /data
fi

# If mode either SLURM or single-node:
if [[ "$MODE" == "slurm" || "$MODE" == "single-node" ]]; then
    # RAY_SESSION_DIR is needed by the Ray head node -> container path
    RAY_SESSION_DIR=$(get_arg_value "--ray_temp_dir" "/tmp/ray/$USER")
    RAY_SESSION_DIR=$(realpath $RAY_SESSION_DIR)
    add_bind $RAY_SESSION_DIR "/tmp/ray"
    set_arg_value "--ray_temp_dir" "/tmp/ray"

    # WORKER_DATA_DIR is needed on the SLURM worker node -> real path
    WORKER_DATA_DIR=$(get_arg_value "--worker_data_dir" $DATA_DIR)
    if [[ -n "$WORKER_DATA_DIR" ]]; then
        WORKER_DATA_DIR=$(realpath $WORKER_DATA_DIR)
        set_arg_value "--worker_data_dir" $WORKER_DATA_DIR
    fi

    # SLURM_LOG_DIR is needed on the SLURM worker node -> real path
    SLURM_LOG_DIR=$(get_arg_value "--slurm_log_dir" "$WORKING_DIR/logs")
    SLURM_LOG_DIR=$(realpath $SLURM_LOG_DIR)
    set_arg_value "--slurm_log_dir" $SLURM_LOG_DIR
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

# Cleanup function
cleanup() {
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

# Run with clean environment
$CONTAINER_CMD exec \
    --cleanenv \
    --pwd /app \
    "${ENV_VARS[@]}" \
    "${BIND_OPTS[@]}" \
    "$IMAGE_PATH" \
    python -m bioengine_worker "${BIOENGINE_WORKER_ARGS[@]}"