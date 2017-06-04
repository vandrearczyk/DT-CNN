import numpy as np
import os
import errno
import argparse


# Make sure that caffe is on the python path:
caffe_root = './'  # this file is expected to be in {caffe_root}/examples
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError, e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e

def main(args):
    # if only test, the caffemodels must be downloaded:
    if  args.solve=="test":
        try:
            open('examples/dyntex++/caffemodels/dyntex++_xy_tcnn_small.caffemodel', 'r')
            open('examples/dyntex++/caffemodels/dyntex++_xt_tcnn_small.caffemodel', 'r')
            open('examples/dyntex++/caffemodels/dyntex++_yt_tcnn_small.caffemodel', 'r')
        except IOError:
            print('The trained caffemodels do not exist. They can be downloaded (and untared) from the url in examples/dyntex++/caffemodels/caffemodels-url')
            return
    
    n_class = 36
    test_iter = 1800 # 
    batch_size = 50 # test batch size
    train_iter = 20000 # train iterations
    acc_xy = np. empty(0)
    acc_xt = np. empty(0)
    acc_yt = np. empty(0)
    acc_xyt = np. empty(0)
    scores = [0]*n_class*test_iter
    labels = [0]*test_iter
    
    
    # ***************************************************************************** XY ************************************************************************************
    # link lmdbs
    symlink_force('dyntex++50_test_lmdb_xy', './examples/dyntex++/dyntex++50_test_lmdb')
    symlink_force('dyntex++50_train_lmdb_xy', './examples/dyntex++/dyntex++50_train_lmdb')
    symlink_force('dyntex++50_mean_xy.binaryproto', './data/dyntex++/dyntex++50_mean.binaryproto')
    
    # Train the network.
    # init
    caffe.set_device(0)
    caffe.set_mode_gpu()
    solver = caffe.SGDSolver('examples/dyntex++/solver_tcnn_small.prototxt')
    
    if args.solve=="train":
        # copy net pre-trained on (resized) ImageNet
        solver.net.copy_from('./models/tcnn/tcnn_small.caffemodel')
        # Train for train_iter iterations 
        solver.step(train_iter)
        print 'network trained xy'
        # share the trained weights 
    else:
        # copy the saved caffemodel
        solver.net.copy_from('examples/dyntex++/caffemodels/dyntex++_xy_tcnn_small.caffemodel')
    solver.test_nets[0].share_with(solver.net)
    
    # Test the trained model on the test set.
    for it in range(test_iter): # batch size (test) = 50, test_iter = 1800 -> 50*1800 = 90000 (50x100x36)/2 testing images
        solver.test_nets[0].forward()
        outlabels = solver.test_nets[0].blobs['ip2bis'].data
        labels[it] = solver.test_nets[0].blobs['label'].data[0]
        scores_xy = [0]*n_class
        for n in range (batch_size):
            for c in range(n_class):
                scores[it*n_class+c] +=  outlabels[n,c]
                scores_xy[c] +=  outlabels[n,c]
        acc_xy = np.append(acc_xy, int(scores_xy.index(max(scores_xy)) == labels[it]))
        
    # ***************************************************************************** XT ************************************************************************************
    # link lmdbs
    symlink_force('dyntex++50_test_lmdb_xt', 'examples/dyntex++/dyntex++50_test_lmdb')
    symlink_force('dyntex++50_train_lmdb_xt', 'examples/dyntex++/dyntex++50_train_lmdb')
    symlink_force('dyntex++50_mean_xt.binaryproto', 'data/dyntex++/dyntex++50_mean.binaryproto')
    
    solver = None
    solver = caffe.SGDSolver('examples/dyntex++/solver_tcnn_small.prototxt')
    if args.solve=="train":
        solver.net.copy_from('./models/tcnn/tcnn_small.caffemodel')
        # Train for train_iter iterations
        solver.step(train_iter)
        print 'network trained xt'
    else:
        # copy the saved caffemodel
        solver.net.copy_from('examples/dyntex++/caffemodels/dyntex++_xt_tcnn_small.caffemodel')
    # share the trained weights 
    solver.test_nets[0].share_with(solver.net)
        
    
    # Test the trained model on the test set.
    for it in range(test_iter): # batch size (test) = 50, test_iter = 1800 -> 50*1800 = 90000 (50x100x36)/2 testing images
        solver.test_nets[0].forward()
        outlabels = solver.test_nets[0].blobs['ip2bis'].data
        scores_xt = [0]*n_class
        
        for n in range (batch_size):
            for c in range(n_class):
                scores[it*n_class+c] = scores[it*n_class+c] +  outlabels[n,c]
                scores_xt[c] = scores_xt[c] +  outlabels[n,c]
        acc_xt = np.append(acc_xt, int(scores_xt.index(max(scores_xt)) == labels[it]))
    
    # ***************************************************************************** YT ************************************************************************************
    # link lmdbs
    symlink_force('dyntex++50_test_lmdb_yt', 'examples/dyntex++/dyntex++50_test_lmdb')
    symlink_force('dyntex++50_train_lmdb_yt', 'examples/dyntex++/dyntex++50_train_lmdb')
    symlink_force('dyntex++50_mean_yt.binaryproto', 'data/dyntex++/dyntex++50_mean.binaryproto')

    solver = None
    solver = caffe.SGDSolver('examples/dyntex++/solver_tcnn_small.prototxt')
    if args.solve=="train":
        solver.net.copy_from('./models/tcnn/tcnn_small.caffemodel')
        # Train for train_iter iterations
        solver.step(train_iter)
        print 'network trained yt'
    else:
        # copy the saved caffemodel
        solver.net.copy_from('examples/dyntex++/caffemodels/dyntex++_yt_tcnn_small.caffemodel')
    # share the trained weights 
    solver.test_nets[0].share_with(solver.net)
        
    
    # Test the trained model on the test set.
    for it in range(test_iter): # batch size (test) = 50, test_iter = 1800 -> 50*1800 = 90000 (50x100x36)/2 testing images
        solver.test_nets[0].forward()
        outlabels = solver.test_nets[0].blobs['ip2bis'].data
        scores_yt = [0]*n_class
        
        for n in range (batch_size):
            for c in range(n_class):
                scores[it*n_class+c] = scores[it*n_class+c] +  outlabels[n,c]
                scores_yt[c] = scores_yt[c] +  outlabels[n,c]
        acc_yt = np.append(acc_yt, int(scores_yt.index(max(scores_yt)) == labels[it]))
        
#*************************** XY + XT + YT ****************************************************************************************************
    for it in range(test_iter):
        score = scores[it*n_class:it*n_class+n_class]
        acc_xyt = np.append(acc_xyt, int(score.index(max(score)) == labels[it]))
    print 'accuracy xy',np.mean(acc_xy)
    print 'accuracy xt',np.mean(acc_xt)
    print 'accuracy yt',np.mean(acc_yt)
    print 'xy + xt + yt accuracy',np.mean(acc_xyt)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dyntex++ classification with the DT-CNN-AlexNet for small input images (Texture-CNN-small on three orthogonal planes)',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--solve', metavar='solve', choices=["test","train"], type=str, default="train",
                        help='train: Train and test the networks; test: Classify using saved caffemodels')
    
    main(parser.parse_args())



