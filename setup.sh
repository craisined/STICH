curl -Z -o data/humtrans.zip -L https://huggingface.co/datasets/dadinghh2/HumTrans/resolve/main/all_wav.zip -o data/musicnet.zip -L https://www.kaggle.com/api/v1/datasets/download/imsparsh/musicnet-dataset/musicnet.npz
unzip data/humtrans.zip -d data/humtrans
rm data/humtrans.zip
unzip data/musicnet.zip -d data/musicnet
rm data/musicnet.zip
mv data/humtrans/wav_data_sync_with_midi/* data/humtrans
rmdir data/humtrans/wav_data_sync_with_midi

pip install -r requirements.txt
python3 data/process_humtrans.py & python3 data/process_musicnet.py 