#!/bin/bash

# https://github.com/Kaggle/kaggle-api
pip install --user kaggle

# https://www.kaggle.com/competitions/tpu-getting-started/data
kaggle competitions download -c tpu-getting-started

unzip tpu-getting-started.zip -d data
rm tpu-getting-started.zip
