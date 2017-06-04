# DT-CNN
Convolutional Neural Network on Three Orthogonal Planes for Dynamic Texture Classification
=======

Implementation of the DT-CNN approach from the following paper:

    Convolutional Neural Network on Three Orthogonal Planes for Dynamic Texture Classification
    Vincent Andrearczyk and Paul F. Whelan
    arXiv:1703.05530

The Caffe master must be installed with pycaffe:
https://github.com/BVLC/caffe


The folders and files in this repository must be added to the master Caffe repository (root-caffe/) as follows.
'root-caffe/examples/dyntex++/'
'root-caffe/examples/solve_dyntex++.py'
'root-caffe/models/tcnn/'
'root-caffe/data/dyntex++/'

The data (lmdb's) must be downloaded and untared from the url provided in 'root-caffe/examples/dyntex++/lmdbs-url'.
The caffemodel (tcnn_small.caffemodel) pre-trained on a resized version of ImageNet must be downloaded and untared from the url
provided in 'root-caffe/models/tcnn/caffemodel-url'.

'solve_dyntex++.py' solves a 50/50 split of the Dyntex++ dataset as described in the paper (result are averaged over multiple splits in the paper) using the DT-CNN-AlexNet for small images.

Once everything is installed and copied, you can run examples/solve_dyntex++.py from the 'root-caffe/' directory as follows:


    usage: solve_dyntex++.py [-h] [--solve solve]
    
    Dyntex++ classification with the DT-CNN-AlexNet for small input images
    (Texture-CNN-small on three orthogonal planes)
    
    optional arguments:
      -h, --help     show this help message and exit
      --solve solve  train: Train and test the networks; test: Classify using saved caffemodels
      

The average results over the test set are printed using single planes and three orthogonal planes.

