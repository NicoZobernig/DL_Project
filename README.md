# DL_Project
Zero-Shot Learning project for Deep Learning course


## Converting the APY dataset to ZSL format

Run the scripts to download necessary data.
```
bash download_apy.sh
bash download_glove.sh
python download_irevnet_pretrained.py
```
Then, run the script to convert the Yahoo dataset to standard format
```
python yahoo_to_zsl.py
```

