#!/bin/bash

set -euxo pipefail

apt-get update

rm -rf /var/lib/apt/lists/*

# install pytorch
pip install pytorch-lightning
pip install matplotlib
pip install open3d
