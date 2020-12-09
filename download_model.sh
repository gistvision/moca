#!/bin/bash
echo "Downloading Pretrained MOCA Weight ..."
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/pretrained.pth
mv pretrained.pth exp/moca

# we provide the pretrained mask rcnn
wget https://alfred-colorswap.s3.us-east-2.amazonaws.com/weight_maskrcnn.pt
echo "done."
