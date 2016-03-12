
import dicom, lmdb, re, sys, cv2
import os, fnmatch, shutil, subprocess
from IPython.utils import io
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore') # we ignore a RuntimeWarning produced from dividing by zero

# .bash_profile changed to include the following path, so this is not needed anymore
CAFFE_ROOT = "/Users/stevenydc/Documents/Caffe\ Installation/caffe_FCN"
# caffe_path = os.path.join(CAFFE_ROOT, "python")
# if caffe_path not in sys.path:
#     print caffe_path
#     sys.path.insert(0, caffe_path)
import caffe
print("\nSuccessfully imported packages, hooray!\n")



'''
===== Instantiate FCN in Caffe =====
'''
from caffe import layers as L
from caffe import params as P

n = caffe.NetSpec()

# helper functions for common structures
def conv_relu(bottom, ks, nout, weight_init='gaussian', weight_std=0.01, bias_value=0, mult=1, stride=1, pad=0, group=1):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         weight_filler=dict(type=weight_init, mean=0.0, std=weight_std),
                         bias_filler=dict(type='constant', value=bias_value),
                         param=[dict(lr_mult=mult, decay_mult=mult), dict(lr_mult=2*mult, decay_mult=0*mult)])
    return conv, L.ReLU(conv, in_place=True)

def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)

def FCN(images_lmdb, labels_lmdb, batch_size, include_acc=False):
    # net definition
    n.data = L.Data(source=images_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=1,
                    transform_param=dict(crop_size=0, mean_value=[0], mirror=False))
    n.label = L.Data(source=labels_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=1)
    n.conv1, n.relu1 = conv_relu(n.data, ks=5, nout=100, stride=1, pad=0, bias_value=0.1)

    n.conv2, n.relu2 = conv_relu(n.conv1, ks=3, nout=200, stride=1, bias_value=0.1)
    n.pool2 = max_pool(n.relu2, ks=2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, ks=3, nout=300, stride=1, bias_value=0.1)
    n.conv4, n.relu4 = conv_relu(n.relu3, ks=3, nout=300, stride=1, bias_value=0.1)
    n.pool4 = max_pool(n.relu4, ks=2, stride=2)
    n.drop4 = L.Dropout(n.pool4, dropout_ratio=0.5, in_place=True)
    n.fcc5 = L.InnerProduct(n.drop4, num_output=1000)
    n.relu5 = L.ReLU(n.fcc5, in_place=True)
    n.fcc6 = L.InnerProduct(n.fcc5, num_output=600)
    # n.score_classes, _= conv_relu(n.drop4, ks=1, nout=2, weight_std=0.01, bias_value=0.1)
    # n.upscore = L.Deconvolution(n.score_classes)
    # n.score = L.Crop(n.upscore,n.data)
    n.loss = L.SigmoidCrossEntropyLoss(n.fcc6, n.label, loss_param=dict(normalize=True))

    # if include_acc:
    #     n.accuracy = L.Accuracy(n.score, n.label)
    #     return n.to_proto()
    # else:
    #     return n.to_proto()
    return n.to_proto()

def make_nets():
    header = 'name: "FCN"\nforce_backward: true\n'
    with open('./TestCaffeNet/train_systole.prototxt', 'w') as f:
        f.write(header + str(FCN('10_img_train_from_csv/', 'db_10img_encoded_label_systole/', batch_size=1, include_acc=False)))
    with open('./TestCaffeNet/test_systole.prototxt', 'w') as f:
        f.write(header + str(FCN('10_img_train_from_csv/', 'db_10img_encoded_label_systole/', batch_size=1, include_acc=True)))

    with open('./TestCaffeNet/train_diastole.prototxt', 'w') as f:
        f.write(header + str(FCN('10_img_train_from_csv/', 'db_10img_encoded_label_diastole/', batch_size=1, include_acc=False)))
    with open('./TestCaffeNet/test_diastole.prototxt', 'w') as f:
        f.write(header + str(FCN('10_img_train_from_csv/', 'db_10img_encoded_label_diastole/', batch_size=1, include_acc=True)))


    #
    # with open('./TestCaffeNet/fcn_test.prototxt', 'w') as f:
    #     f.write(header + str(FCN('val_images_lmdb/', 'val_labels_lmdb/', batch_size=1, include_acc=True)))

# make_nets()




# caffe.set_mode_cpu()

os.chdir('./TestCaffeNet')
try:
    solver = caffe.SGDSolver('./solver_systole.prototxt')
except:
    print "oh well."

#Run this command in shell:
'''
/Users/stevenydc/Documents/Caffe\ Installation/caffe_FCN/build/tools/caffe train -solver=solver_systole.prototxt > systole_test.log
'''


'''
===== Prediction! =====
'''
caffe.set_mode_cpu()
net = caffe.Net('./TestCaffeNet/deploy_systole.prototxt', './TestCaffeNet/model_logs/systole_sigmoid_iter_150.caffemodel', caffe.TEST)
img = net.blobs['data'].data
# for i in range(3):
#     plt.figure()
#     plt.imshow(img[0,4*i,...])
# plt.imshow(img[0,0,...])
systole_prob = []
for i in range(50):
    net.forward()

    pred = net.blobs['prob'].data.ravel()
    plt.plot(pred)
    systole_prob.append(pred)




'''
===== Calculate accuracy =====
'''
import csv
def accumulate_result(validate_lst, prob):
    sum_result = {}
    cnt_result = {}
    size = prob.shape[0]
    fi = csv.reader(open(validate_lst))
    for i in range(size):
        line = fi.next() # Python2: line = fi.next()
        idx = int(line[0])
        if idx not in cnt_result:
            cnt_result[idx] = 0.
            sum_result[idx] = np.zeros((1, prob.shape[1]))
        cnt_result[idx] += 1
        sum_result[idx] += prob[i, :]
    for i in cnt_result.keys():
        sum_result[i][:] /= cnt_result[i]
    return sum_result

systole_result = accumulate_result("./Proto-1-train-label.csv", np.array(systole_prob))
# diastole_result = accumulate_result("./Proto-1-train-label.csv", diastole_prob)

Proto_valid_label_systole = np.loadtxt('Proto-1-validate-label-systole-encoded.csv', delimiter=',')
for idx, key in enumerate(systole_result.keys()):
    print np.sum(np.abs(Proto_valid_label_systole[idx]))
    # print np.sum(np.abs(systole_result[key]-Proto_valid_label_systole[idx]))




