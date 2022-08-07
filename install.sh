git clone https://github.com/clovaai/CRAFT-pytorch.git
git clone https://github.com/clovaai/deep-text-recognition-benchmark

# create dir for models
mkdir weights
# craft main
wget -O weights/craft_mlt_25k.pth https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ&export=download
# craft refiner
wget -O weights/craft_refiner_CTW1500.pth https://drive.google.com/uc?id=1XSaFwBkOaFOdtk4Ane3DFyJGPRw6v5bO&export=download
# deep text recongnition models
wget -O weights/TPS-ResNet-BiLSTM-Attn.pth https://www.dropbox.com/sh/j3xmli4di1zuv3s/AADbTu4LF-nMUBmC43_RQ8OGa/TPS-ResNet-BiLSTM-Attn.pth?dl=1
wget -O weights/TPS-ResNet-BiLSTM-CTC.pth https://www.dropbox.com/sh/j3xmli4di1zuv3s/AAB0X-sX05-0psb4uXWPYSmza/TPS-ResNet-BiLSTM-CTC.pth?dl=1

#Please Read man https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
#wget -O weights/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth https://drive.google.com/uc?id=1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY&export=download&confirm=t
gdown --id 1ajONZOgiG9pEYsQ-eBmgkVbMDuHgPCaY -O weights/TPS-ResNet-BiLSTM-Attn-case-sensitive.pth 


# Result images folders, probably it can, be not necessary
mkdir ./results_images/

echo ">>>>>I install all.<<<<<"
echo "Don't forget run command for installing libs \"pil install -r requirements.txt\""