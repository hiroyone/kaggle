# Install kaggle api in advance
# pip install kaggle --upgrade
# Make a directory for Kaggle if not
# mkdir Kaggle && cd Kaggle
# Make a directory for a competition
mkdir ga-customer-revenue-prediction && cd $_
# Make a directory for input folder
mkdir input && cd $_
# Download all inputs for the competition
kaggle competitions download -c ga-customer-revenue-prediction
# Unpacking all compressed data
# gunzip *.gz 
cd ..
# Install sample kernels
mkdir kernels && cd $_
# 1. https://kaggle.com/sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
kaggle kernels pull sudalairajkumar/simple-exploration-baseline-ga-customer-revenue
cd ..