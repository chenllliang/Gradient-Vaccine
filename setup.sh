pip install torch==1.11.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu115 # cu114,115
pip install portalocker==1.0.0
pip install tensorboardX
pip install jupyter
pip install tensorboard

# for windows user, should be in adminitrator mode and install visual studio .

cd fairseq
pip install --editable ./
pip install sacremoses
cd ..