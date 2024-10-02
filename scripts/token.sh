#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
SCRIPT_PATH="$SCRIPT_DIR/../hypha/token_init.py"
output=$(python -u "$SCRIPT_PATH" | tee /dev/tty)

# Check if the last line of output matches the regex for "export x=y"
if [[ $output =~ (export\ [a-zA-Z_][a-zA-Z0-9_]*=[^ ]+) ]]; then
    export_variable="${BASH_REMATCH[1]}"
    echo "Output is valid: $export_variable"
    eval "$export_variable"
else
    echo "FAIL"
    exit 1
fi
