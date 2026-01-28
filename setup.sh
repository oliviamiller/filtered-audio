#!/bin/sh

set -eu

cd `dirname $0`

# Install unzip if not present (needed for Vosk model extraction)
if ! command -v unzip >/dev/null; then
    echo "Installing unzip..."
    if command -v apt-get >/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y unzip curl
    fi
fi

# Create a virtual environment to run our code
VENV_NAME="venv"
PYTHON="$VENV_NAME/bin/python"

# Try to create virtualenv
if ! python3 -m venv $VENV_NAME 2>&1; then
    echo "Failed to create virtualenv. Installing python3, python3-pip, and python3-venv..."
    if command -v apt-get >/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv
        if ! python3 -m venv $VENV_NAME 2>&1; then
            echo "Failed to create virtualenv even after installing dependencies" >&2
            exit 1
        fi
    else
        echo "This module requires Python >=3.8, pip, and virtualenv to be installed." >&2
        exit 1
    fi
fi

# remove -U if viam-sdk should not be upgraded whenever possible
# -qq suppresses extraneous output from pip
echo "Virtualenv found/created. Installing/upgrading Python packages..."
if ! [ -f .installed ]; then
     if ! $PYTHON -m pip install --upgrade pip setuptools wheel; then
        echo "Failed to upgrade pip tooling" >&2
        exit 1
    fi

    if ! $PYTHON -m pip install --only-binary=:all: -r requirements.txt; then
        echo "Failed to install Python packages" >&2
        exit 1
    fi
    touch .installed

fi
