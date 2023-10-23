#!/bin/bash

# You have to install kaggle API first
# https://github.com/Kaggle/kaggle-api#installation
# For example: pip install --user kaggle

# Next, get your API credentials from Kaggle
# https://github.com/Kaggle/kaggle-api#api-credentials
# Go to the 'Account' tab of your user profile (https://www.kaggle.com/<username>/account)
# and select 'Create API Token'. This will trigger the download of kaggle.json, a file containing your API credentials.
# Place this file in the location ~/.kaggle/kaggle.json

# Now, you can download the data from Kaggle
# https://www.kaggle.com/competitions/tpu-getting-started/data
kaggle competitions download -c tpu-getting-started

# Unzip the data
unzip tpu-getting-started.zip -d data
rm tpu-getting-started.zip
