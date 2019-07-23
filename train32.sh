#!/bin/bash

#set -x
set -e

if [ $# -ne 2 ] ; then
    echo 'Please input parameters ROOT_FOLDER and GPU_ID'
    exit 1;
fi

CAFFE_ROOT=/home/zl/caffe
ROOT_FOLDER=$1  # image root folder
GPU_ID=$2

# prepare  sudo ./train32.sh ./flickr_25 1

# prepare
binarytestPath="./analysis/32/binary-test.mat"
if [ ! -f "$binarytestPath" ]; then
    echo 'binarytestPath OK!'
else
    rm -r ./analysis/32/binary-test.mat
    echo 'binarytestPath Delete!'    
fi

binarytraintPath="./analysis/32/binary-train.mat"
if [ ! -f "$binarytraintPath" ]; then
    echo 'binarytraintPath OK!'
else
    rm -r ./analysis/32/binary-train.mat
    echo 'binarytraintPath Delete!'
fi

feattraintPath="./analysis/32/feat-train.mat"
if [ ! -f "$feattraintPath" ]; then
    echo 'feattraintPath OK!'
else
    rm -r ./analysis/32/feat-train.mat
    echo 'feattraintPath Delete!'
fi

feattestPath="./analysis/32/feat-test.mat"
if [ ! -f "$feattestPath" ]; then
    echo 'feattestPath OK!'
else
    rm -r ./analysis/32/feat-test.mat
    echo 'feattestPath Delete!'
fi

B32="./data_from_DWDH/B_32bits.h5"
if [ ! -f "$B32" ]; then
    echo 'B32 OK!'
else
    rm -r ./data_from_DWDH/B_32bits.h5
    echo 'B32 Delete!'
fi

#rm -r ./analysis/32/binary-test.mat
#rm -r ./analysis/32/binary-train.mat
#rm -r ./analysis/32/feat-test.mat
#rm -r ./analysis/32/feat-train.mat
#rm -r ./data_from_DWDH/B_32*
#rm -r ./fc7_features/traindata_32*.txt

# iteration 1
#echo "iteration 1"
#echo "extract fc7 features"
#cd fc7_features
#python extract_features32.py vgg_16 ../caffemodels/VGG_ILSVRC_16_layers.caffemodel ../$ROOT_FOLDER ../flickr_25/train_file_list.txt fc7 traindata_32
#cd ..
#echo "generate .mat file -> ./fc7_features/traindata_32.txt"

#echo "update anchors by DWDH algorithm"
export PATH=$PATH:/usr/local/MATLAB/R2016b/bin

matlab -nojvm -nodesktop -r "run ./DWDH/DWDH_32.m; quit;"
echo "generate .h5 file -> ./data_from_DWDH/B_32bits.h5"

echo "finetune VGG model to initialize W."
cd finetune_network
$CAFFE_ROOT/build/tools/caffe train -solver ./solver_32bits.prototxt -weights ../caffemodels/VGG_ILSVRC_16_layers.caffemodel -gpu $GPU_ID
cd ..
echo "finetuning finished!"

echo "test"
matlab -nojvm -nodesktop -r "run ./run_flickr25_32bits.m; quit;"
echo "finished!"
