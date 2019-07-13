#!/usr/bin/env bash

mkdir files
mkdir files/data

wget -c "https://www.dropbox.com/s/3m3tzxhffvjivj6/train.zip?dl=0"
mv "train.zip?dl=0" files/data/train.zip
unzip files/data/train.zip -d files/data/

cp $1 files/we.vec