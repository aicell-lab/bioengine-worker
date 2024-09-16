#!/bin/bash

LOGS_DIR="$HOME/logs"
if [ ! -d "$LOGS_DIR" ]; then
    mkdir -p "$LOGS_DIR"
    echo "Created logs directory at $LOGS_DIR"
else
    echo "Logs directory already exists at $LOGS_DIR"
fi
