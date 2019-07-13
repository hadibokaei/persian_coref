#!/usr/bin/env bash

rm -rf files

mkdir files
mkdir files/data

wget -c "https://www.dropbox.com/s/3m3tzxhffvjivj6/train.zip?dl=0"
mv "train.zip?dl=0" files/data/train.zip
unzip files/data/train.zip -d files/data/

pwd
echo 1
cp $1 files/we.vec
echo 2
python script_prepare_data.py
python script_train.py