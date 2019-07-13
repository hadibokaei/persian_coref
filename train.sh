#!/usr/bin/env bash

mkdir files
mkdir files/data

wget -c "https://www.dropbox.com/s/3m3tzxhffvjivj6/train.zip?dl=0"
mv "train.zip?dl=0" files/data/train.zip
unzip train.zip