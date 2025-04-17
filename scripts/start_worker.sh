# !/bin/bash

VERSION=0.1.4
WORKING_DIR=$(pwd)


# Save all arguments
BIOENGINE_WORKER_ARGS=("$@")

# Check if Apptainer is installed
if ! command -v apptainer &> /dev/null
then
    echo "Apptainer could not be found. Please install it first."
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
APPTAINER_IMAGE_PATH="$(get_arg_value "--image_path" "$WORKING_DIR/apptainer_images/bioengine-worker_$VERSION.sif")"
APPTAINER_IMAGE_PATH=$(realpath $APPTAINER_IMAGE_PATH)
set_arg_value "--image_path" $APPTAINER_IMAGE_PATH

# Set the directory and image name
APPTAINER_IMAGE_DIR=$(dirname "$APPTAINER_IMAGE_PATH")
APPTAINER_IMAGE=$(basename "$APPTAINER_IMAGE_PATH")

# Check if the BioEngine worker image is available
if [ ! -f "$APPTAINER_IMAGE_PATH" ]; then
    
    VERSION=${APPTAINER_IMAGE#bioengine-worker_}
    VERSION=${VERSION%.sif}
    DOCKER_IMAGE=bioengine-worker:$VERSION
    # Ask user before pulling
    read -p "Image file $APPTAINER_IMAGE_PATH not found. Do you want to pull it from Docker? (yes/no): " response
    if [[ "$response" =~ ^[Yy][Ee][Ss]$ ]]; then
        echo "Pulling image $DOCKER_IMAGE from Docker..."
        mkdir -p $APPTAINER_IMAGE_DIR
        apptainer pull $APPTAINER_IMAGE_PATH docker://ghcr.io/aicell-lab/$DOCKER_IMAGE
        if [[ $? -eq 0 ]]; then
            echo "Image pulled successfully to $APPTAINER_IMAGE_PATH"
        else
            echo "Error pulling image"
            exit 1
        fi
    else
        echo "Image download cancelled"
        exit 1
    fi
fi


# Define environment variables
ENV_VARS=()
add_env() {
    if [[ -n "$2" ]]; then
        ENV_VARS+=("--env=$1=$2")
    fi
}

# Export environment variables from .env file
set -a
source $WORKING_DIR/.env
set +a

# Add environment variables
add_env "USER" "$USER"
add_env "HYPHA_TOKEN" "$HYPHA_TOKEN"


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

# Add SLURM-specific bindings

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


# Add BioEngine worker bindings

# RAY_SESSION_DIR is needed by the Ray head node -> container path
RAY_SESSION_DIR=$(get_arg_value "--ray_temp_dir" "/tmp/ray/$USER")
RAY_SESSION_DIR=$(realpath $RAY_SESSION_DIR)
mkdir -p "$RAY_SESSION_DIR"
chmod 777 $RAY_SESSION_DIR
add_bind $RAY_SESSION_DIR "/tmp/ray"
set_arg_value "--ray_temp_dir" "/tmp/ray"  

# SLURM_LOGS_DIR is needed on the SLURM worker node -> real path
SLURM_LOGS_DIR=$(get_arg_value "--slurm_logs" "$WORKING_DIR/slurm_logs")
SLURM_LOGS_DIR=$(realpath $SLURM_LOGS_DIR)
mkdir -p "$SLURM_LOGS_DIR"
chmod 777 $SLURM_LOGS_DIR
set_arg_value "--slurm_logs" $SLURM_LOGS_DIR

# DATA_DIR is needed on the SLURM worker node -> real path
DATA_DIR=$(get_arg_value "--data_dir" "")
if [[ -z "$DATA_DIR" ]]; then
    echo "Error: the following argument is required: --data_dir"
    exit 1
fi
DATA_DIR=$(realpath $DATA_DIR)
mkdir -p "$DATA_DIR"
# Don't set permissions for data directory
add_bind $DATA_DIR "/data" "ro"  # Read-only
set_arg_value "--data_dir" $DATA_DIR


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

# Run with clean environment
apptainer exec \
    --cleanenv \
    --pwd /app \
    ${ENV_VARS[@]} \
    ${BIND_OPTS[@]} \
    "$APPTAINER_IMAGE_PATH" \
    python -m bioengine_worker "${BIOENGINE_WORKER_ARGS[@]}"
    