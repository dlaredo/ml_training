#!/bin/bash
KEY=$(jq .preprocess_data config.json)

if [[ $KEY = '"True"' ]]; then
  python data_preprocess.py
fi

if [[ $? -eq 0 ]]; then
  python train_model.py
else
  echo "Stopping execution"
fi

