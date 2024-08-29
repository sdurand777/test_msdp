# go to the right place
cd /home/smith/test_msdp


# download models
curl -L -o vo.pth "https://drive.usercontent.google.com/download?id=14c8KqszsLuT9ZqS7OtSkfHBZ030dyDKP&export=download&authuser=0&confirm=t&uuid=1d9448ff-5602-47ce-8aa5-1821d8db893d&at=AO7h07ey-Xxd3OtwUdYCvvfZ_-9C:1724676041859"
curl -L -o twoview.pth "https://drive.usercontent.google.com/download?id=1qibBGQqwAC06QkC6VapOcFJWVWhvnRgG&export=download&authuser=0&confirm=t&uuid=c93c9279-01e5-4879-b09e-be9937368e3e&at=AO7h07fqhxhiOSdDMTXdBHVr_0Dx:1724676615583"
curl -L -o homog_pretrain.pth "https://drive.usercontent.google.com/download?id=1RobrwH6FiRopqVGAy66yQ0e45x-SR3M0&export=download&authuser=0&confirm=t&uuid=c9a7db13-3302-4b7c-afd9-716b19ffdfcf&at=AO7h07fnnZb10FADcKTF-n-YL8xY:1724676621850"

# download netvlad
mkdir -p /home/smith/.cache/torch/hub/netvlad/
wget https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat -O /home/smith/.cache/torch/hub/netvlad/VGG16-NetVLAD-Pitts30K.mat

