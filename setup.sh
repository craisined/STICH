curl -Z -o data/humtrans.zip -L https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/all_wav.zip -o data/musicnet.zip -L https://www.kaggle.com/api/v1/datasets/download/imsparsh/musicnet-dataset
unzip data/humtrans.zip -d data/humtrans
unzip data/musicnet.zip -d data/musicnet
mv data/humtrans/wav_data_sync_with_midi/* data/humtrans
rmdir data/humtrans/wav_data_sync_with_midi