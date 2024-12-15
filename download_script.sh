#!/bin/bash


output_folder="data"

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1phfeL1utZSlp57JqRIS9m483riKNuY4B' -O data.zip

# Unzip the files and remove the .zip files
unzip data.zip
rm data.zip