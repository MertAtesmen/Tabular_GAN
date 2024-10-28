#!/bin/bash

# Datasets the project will download from kaggle
datasets=("creditcardfraud" "30000-spotify-songs")
users=("mlg-ulb" "joebeachcapital")
output_folder="data"
kaggle_api="https://www.kaggle.com/api/v1/datasets/download"

n_datasets=${#datasets[@]}
number_of_connections=1

# Check if axel is present and use if available
if command -v axel > /dev/null; then
    for ((i=0; i<n_datasets; i++)); do
        axel -n $number_of_connections -o "$output_folder/${datasets[$i]}.zip" "$kaggle_api/${users[$i]}/${datasets[$i]}"
    done
else
    for ((i=0; i<n_datasets; i++)); do
        curl -L -o "$output_folder/${datasets[$i]}.zip" "$kaggle_api/${users[$i]}/${datasets[$i]}"
    done
fi

# Unzip the files and remove the .zip files
unzip "$output_folder/*.zip" -d "$output_folder/"
rm $output_folder/*.zip $output_folder/*.md