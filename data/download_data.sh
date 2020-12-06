#!/bin/bash

# Install 7z
echo "Checking for 7z (might require installation)..."

if sudo apt-get install p7zip-full -y; then
    echo "7z found/installed"
else
    echo "Failed: Please install 7z (https://www.7-zip.org/7z.html)"
    exit 1
fi

# Download, Unzip, and Remove zip
echo "Downloading JSONs and Resnet18 feats ..."

wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/json_feat_2.1.0.7z

7z x json_feat_2.1.0.7z -y && rm json_feat_2.1.0.7z
echo "saved folder: json_feat_2.1.0"

