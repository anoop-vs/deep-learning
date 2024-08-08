#!/bin/bash

url="http://nlp.stanford.edu/data/glove.6B.zip"
filename="glove.6B.zip"
target_directory="artifacts/LSTM/"
mkdir -p "$target_directory"
cd "$target_directory"
wget "$url"
unzip "$filename"
rm "$filename"
