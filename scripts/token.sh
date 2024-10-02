#!/bin/bash

echo "hello test"

SCRIPT_DIR=$(dirname "$0")
SCRIPT_PATH="$SCRIPT_DIR/../hypha/token_init.py"

python "$SCRIPT_PATH"

# Check if the output matches the regex for "export x=y"
if [[ $output =~ ^export\ [a-zA-Z_][a-zA-Z0-9_]*=.+$ ]]; then
    echo "Output is valid: $output"
    # You can optionally evaluate the output to set the environment variable
    eval "$output"
else
    echo "FAIL"
    exit 1  # Return a non-zero exit status to indicate failure
fi
