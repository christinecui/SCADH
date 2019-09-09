# SCADH
Scalable Deep Hashing for Large-scale Social Image Retrieval

Hui Cui, Lei Zhu, Jingjing Li, Yang Yang, Liqiang Nie

The paper has been received by IEEE Transactions on Image Processing, doi: 10.1109/TIP.2019.2940693.


·Prerequisites
1.	Requirements for Caffe, pycaffe and matcaffe.
2.	VGG-16 pre-trained model on ILSVC12 datasets, and save it in caffemodels directory.


·Installation

Enter caffe directory and download the source codes.

    cd caffe/
    
Modify Makefile.config and build Caffe with following commands:

    make all -j8
    
    make pycaffe
    
    make matcaffe
    
    
    
·Usage

We only supply the code to train 32-bit SCADH on MIR Flickr dataset.

We integrate train step and test step in a bash file train32.sh, please run it as follows:

    sudo./train32.sh [ROOT_FOLDER] [GPU_ID]
    
    # ROOT_FOLDER is the root folder of image datasets,
    
    # GPU_ID is the GPU you want to train on,
    
    # e.g. sudo ./train32.sh ./flickr_25 1
    
  
  
·Citation

If you find our approach useful in your research, please consider citing:

@article{'SCADH',

    author   = {Hui Cui and Lei Zhu and Jingjing Li and Yang Yang and Liqiang Nie},
    
    journal  = {IEEE Transactions on Image Processing (TIP)}, 
    
    title    = {Scalable Deep Hashing for Large-scale Social Image Retrieval},
    
    year     = {2019}
    
}



