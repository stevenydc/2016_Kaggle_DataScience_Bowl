
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
===== Prediction! =====
'''
caffe.set_mode_cpu()
# caffe.set_mode_gpu() # Use this line on EC2
systole_net = caffe.Net('./TestCaffeNet/deploy_systole.prototxt', './TestCaffeNet/model_logs/systole_sigmoid_Fcc_drop_Fcc_iter_150.caffemodel', caffe.TEST)
# img = net.blobs['data'].data
# for i in range(3):
#     plt.figure()
#     plt.imshow(img[0,4*i,...])
# plt.imshow(img[0,0,...])

TEST_SET_SIZE = 50

systole_prob = []
for i in range(TEST_SET_SIZE):
    systole_net.forward()

    pred = systole_net.blobs['prob'].data.ravel()
    # plt.plot(pred)
    systole_prob.append(pred)


diastole_net = caffe.Net('./TestCaffeNet/deploy_systole.prototxt', './TestCaffeNet/model_logs/diastole_sigmoid_Fcc_drop_Fcc_iter_150.caffemodel', caffe.TEST)

diastole_prob = []
for i in range(TEST_SET_SIZE):
    diastole_net.forward()

    pred = net.blobs['prob'].data.ravel()
    plt.plot(pred)
    diastole_prob.append(pred)




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


def CRPS(label, pred):
    """ Custom evaluation metric on CRPS.
    """
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1] - 1):
            if pred[i, j] > pred[i, j + 1]:
                pred[i, j + 1] = pred[i, j]
    return np.sum(np.square(label - pred)) / label.size


systole_result = accumulate_result("./Proto-1-train-label.csv", np.array(systole_prob))
diastole_result = accumulate_result("./Proto-1-train-label.csv", np.array(diastole_prob))
# diastole_result = accumulate_result("./Proto-1-train-label.csv", diastole_prob)

'''Following chunk is for local CPU prototype only'''
Proto_valid_label_systole = np.loadtxt('Proto-1-validate-label-systole-encoded.csv', delimiter=',')
kaggle_score = []
for idx, key in enumerate(systole_result.keys()):
    temp = CRPS(Proto_valid_label_systole[idx], systole_result[key])
    kaggle_score.append(temp)
    print np.sum(np.abs(Proto_valid_label_systole[idx]-systole_result[key])),'\t', temp
    # print np.sum(np.abs(systole_result[key]-Proto_valid_label_systole[idx]))


def submission_helper(pred):
    p = np.zeros(600)
    pred.resize(p.shape)
    p[0] = pred[0]
    for j in range(1, 600):
        a = p[j - 1]
        b = pred[j]
        # if b < 0.2:
        # 	p[j] = 0
        # 	continue
        if b < a:
            p[j] = a
        else:
            p[j] = b
    return p


def doHist(data):
    h = np.zeros(600)
    for j in np.ceil(data).astype(int):
        h[j:] += 1
    h /= len(data)
    return h
train_csv = np.genfromtxt("./Data/train-label.csv", delimiter=',')
hSystole = doHist(train_csv[:, 1])
hDiastole = doHist(train_csv[:, 2])


fi = csv.reader(open("./Data/sample_submission_validate.csv"))
f = open("submission.csv", "w")
fo = csv.writer(f, lineterminator='\n')
fo.writerow(fi.next()) # Python2: fo.writerow(fi.next())
for line in fi:
    idx = line[0]
    key, target = idx.split('_')
    key = int(key)
    out = [idx]
    if key in systole_result:
        if target == 'Diastole':
            out.extend(list(submission_helper(diastole_result[key])))
        else:
            out.extend(list(submission_helper(systole_result[key])))
    else:
        print("Miss: %s" % idx)
        if target == 'Diastole':
            out.extend(hDiastole)
        else:
            out.extend(hSystole)
    fo.writerow(out)
f.close()









