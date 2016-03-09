__author__ = 'stevenydc'
'''
===== Loading Libraries =====
'''
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
===== End of Loading Libraries =====
'''


'''
===== Formatting data sets =====
'''
SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-40": "0944",
}

SUNNYBROOK_ROOT_PATH = "./Data/Sunnybrook_data/"

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                            "Sunnybrook Cardiac MR Database ContoursPart3",
                            "TrainingDataContours")
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "challenge_training")

def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))

    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)

    __repr__ = __str__

def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label

def get_all_contours(contour_path):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')]
    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    extracted = map(Contour, contours)
    return extracted

def export_all_contours(contours, img_path, lmdb_img_name, lmdb_label_name):
    for lmdb_name in [lmdb_img_name, lmdb_label_name]:
        db_path = os.path.abspath(lmdb_name)
        if os.path.exists(db_path):
            shutil.rmtree(db_path)
    counter_img = 0
    counter_label = 0
    batchsz = 100
    print("Processing {:d} images and labels...".format(len(contours)))
    for i in xrange(int(np.ceil(len(contours) / float(batchsz)))):
        batch = contours[(batchsz*i):(batchsz*(i+1))]
        if len(batch) == 0:
            break
        imgs, labels = [], []
        for idx,ctr in enumerate(batch):
            try:
                img, label = load_contour(ctr, img_path)  # HERE IS THE REAL DATA COMING IN
                imgs.append(img)
                labels.append(label)
                if idx % 20 == 0:
                    print ctr
                    plt.imshow(img)
                    plt.show()
                    plt.imshow(label)
                    plt.show()
            except IOError:
                continue
        db_imgs = lmdb.open(lmdb_img_name, map_size=1e12)
        with db_imgs.begin(write=True) as txn_img:
            for img in imgs:
                datum = caffe.io.array_to_datum(np.expand_dims(img, axis=0))
                # {:0>10d} is a 10 digit number with 0 paddings on the left
                txn_img.put("{:0>10d}".format(counter_img), datum.SerializeToString())
                counter_img += 1
        print("Processed {:d} images".format(counter_img))
        db_labels = lmdb.open(lmdb_label_name, map_size=1e12)
        with db_labels.begin(write=True) as txn_label:
            for lbl in labels:
                datum = caffe.io.array_to_datum(np.expand_dims(lbl, axis=0))
                txn_label.put("{:0>10d}".format(counter_label), datum.SerializeToString())
                counter_label += 1
        print("Processed {:d} labels".format(counter_label))
    db_imgs.close()
    db_labels.close()


'''
===== End of Formatting data sets =====
'''

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
                    transform_param=dict(crop_size=0, mean_value=[77], mirror=False))
    n.label = L.Data(source=labels_lmdb, backend=P.Data.LMDB, batch_size=batch_size, ntop=1)
    n.conv1, n.relu1 = conv_relu(n.data, ks=5, nout=100, stride=2, pad=50, bias_value=0.1)
    n.pool1 = max_pool(n.relu1, ks=2, stride=2)
    n.conv2, n.relu2 = conv_relu(n.pool1, ks=5, nout=200, stride=2, bias_value=0.1)
    n.pool2 = max_pool(n.relu2, ks=2, stride=2)
    n.conv3, n.relu3 = conv_relu(n.pool2, ks=3, nout=300, stride=1, bias_value=0.1)
    n.conv4, n.relu4 = conv_relu(n.relu3, ks=3, nout=300, stride=1, bias_value=0.1)
    n.drop = L.Dropout(n.relu4, dropout_ratio=0.1, in_place=True)
    n.score_classes, _= conv_relu(n.drop, ks=1, nout=2, weight_std=0.01, bias_value=0.1)
    n.upscore = L.Deconvolution(n.score_classes)
    n.score = L.Crop(n.upscore,n.data)
    n.loss = L.SoftmaxWithLoss(n.score, n.label, loss_param=dict(normalize=True))

    if include_acc:
        n.accuracy = L.Accuracy(n.score, n.label)
        return n.to_proto()
    else:
        return n.to_proto()

def make_nets():
    header = 'name: "FCN"\nforce_backward: true\n'
    with open('fcn_train.prototxt', 'w') as f:
        f.write(header + str(FCN('train_images_lmdb/', 'train_labels_lmdb/', batch_size=1, include_acc=False)))
    with open('fcn_test.prototxt', 'w') as f:
        f.write(header + str(FCN('val_images_lmdb/', 'val_labels_lmdb/', batch_size=1, include_acc=True)))
'''
===== End of Instantiate FCN in Caffe =====
'''


'''TESTING'''
ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
img, label = load_contour(ctrs[0], TRAIN_IMG_PATH)




if __name__== "__main__":
    SPLIT_RATIO = 0.1
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
    val_ctrs = ctrs[0:int(SPLIT_RATIO*len(ctrs))]
    train_ctrs = ctrs[int(SPLIT_RATIO*len(ctrs)):]


    print("Done mapping ground truth contours to images")
    print("\nBuilding LMDB for train...")

    export_all_contours(train_ctrs, TRAIN_IMG_PATH, "train_images_lmdb", "train_labels_lmdb")
    print("\nBuilding LMDB for val...")
    export_all_contours(val_ctrs, TRAIN_IMG_PATH, "val_images_lmdb", "val_labels_lmdb")


    make_nets()

caffe.set_mode_cpu() # or caffe.set_mode_cpu() for machines without a GPU
try:
    del solver # it is a good idea to delete the solver object to free up memory before instantiating another one
    solver = caffe.SGDSolver('fcn_solver.prototxt')
except NameError:
    solver = caffe.SGDSolver('fcn_solver.prototxt')


[(k, v.data.shape) for k, v in solver.net.blobs.items()]





'''
===== Before solving nets, let's visualize some things =====
'''


# each blob has dimensions batch_size x channel_dim x height x width
[(k, v.data.shape) for k, v in solver.net.blobs.items()]

# print the layers with learnable weights and their dimensions
[(k, v[0].data.shape) for k, v in solver.net.params.items()]

# print the biases associated with the weights
[(k, v[1].data.shape) for k, v in solver.net.params.items()]

# params and diffs have the same dimensions
[(k, v[0].diff.shape) for k, v in solver.net.params.items()]


# forward pass with randomly initialized weights
solver.net.forward()  # train net
solver.test_nets[0].forward()  # test net (more than one net is supported)

# visualize the image data and its correpsonding label from the train net
img_train = solver.net.blobs['data'].data[0,0,...]
plt.imshow(img_train)
plt.show()

label_train = solver.net.blobs['label'].data[0,...]
plt.imshow(label_train)
plt.show()

img_train = solver.net.blobs['upscore'].data[0,1,...]
plt.imshow(img_train)
plt.show()



# visualize the image data and its correpsonding label from the test net
img_test = solver.test_nets[0].blobs['data'].data[0,0,...]
plt.imshow(img_test)
plt.show()
label_test = solver.test_nets[0].blobs['label'].data[0,0,...]
plt.imshow(label_test)
plt.show()



# take one step of stochastic gradient descent consisting of both forward pass and backprop
solver.step(1)

# visualize gradients after backprop. If non-zero, then gradients are properly propagating and the nets are learning something
# gradients are shown here as 10 x 10 grid of 5 x 5 filters
plt.imshow(solver.net.params['conv1'][0].diff[:,0,...].reshape(10,10,5,5).transpose(0,2,1,3).reshape(10*5,10*5), 'gray')
plt.show()


plt.imshow(solver.net.params['conv3'][0].diff[:,0,...].reshape(25,12,3,3).transpose(0,2,1,3).reshape(25*3,12*3), 'gray')
plt.show()




'''
===== Training! =====
'''
ret = subprocess.call(os.path.join(CAFFE_ROOT, 'build/tools/caffe') + ' ' +
                      'train -solver=fcn_solver.prototxt > fcn_train.log', shell=True)
# Now we should have the snapshot file of the network at the end of training
# We can use that to do our prediction.
# In other words, we can close the program now?
'''
===== End of Training! =====
'''



'''
===== Making Predictions =====
'''

class Dataset(object):
    # This class is primarily transforming dicom data into useful format
    dataset_count = 0

    def __init__(self, directory, subdir):
        # deal with any intervening directories
        while True:
            subdirs = next(os.walk(directory))[1]
            if len(subdirs) == 1:
                directory = os.path.join(directory, subdirs[0])
            else:
                break

        slices = [] # Basically a collection of indexes for sax folders
        for s in subdirs:
            m = re.match('sax_(\d+)', s)
            if m is not None:
                slices.append(int(m.group(1))) # m.group(1) = string matched with the 1st subgroup (\d+)

        slices_map = {}
        first = True
        times = []
        for s in slices:
            # returns the file names of sax_%d folder
            files = next(os.walk(os.path.join(directory, 'sax_%d' % s)))[2]
            offset = None

            for f in files:
                m = re.match('IM-(\d{4,})-(\d{4})\.dcm', f)
                if m is not None:
                    if first:
                        times.append(int(m.group(2)))
                    if offset is None:
                        offset = int(m.group(1))

            first = False
            slices_map[s] = offset

        self.directory = directory
        self.time = sorted(times)
        self.slices = sorted(slices)
        self.slices_map = slices_map
        Dataset.dataset_count += 1
        self.name = subdir

    def _filename(self, s, t):
        return os.path.join(self.directory,
                            'sax_%d' % s,
                            'IM-%04d-%04d.dcm' % (self.slices_map[s], t))

    def _read_dicom_image(self, filename):
        d = dicom.read_file(filename)
        img = d.pixel_array.astype('int')
        return img

    def _read_all_dicom_images(self):
        f1 = self._filename(self.slices[0], self.time[0])
        d1 = dicom.read_file(f1)
        (x, y) = d1.PixelSpacing
        (x, y) = (float(x), float(y))
        f2 = self._filename(self.slices[1], self.time[0])
        d2 = dicom.read_file(f2)

        # try a couple of things to measure distance between slices
        try:
            dist = np.abs(d2.SliceLocation - d1.SliceLocation)
        except AttributeError:
            try:
                dist = d1.SliceThickness
            except AttributeError:
                dist = 8  # better than nothing...

        self.images = np.array([[self._read_dicom_image(self._filename(d, i))
                                 for i in self.time]
                                for d in self.slices])
        self.dist = dist
        self.area_multiplier = x * y

    def load(self):
        self._read_all_dicom_images()


# Helper function and function to perform LV segmentation and EF calculation

MEAN_VALUE = 77
THRESH = 0.5

def calc_all_areas(images):
    (num_images, times, _, _) = images.shape

    all_masks = [{} for i in range(times)]
    all_areas = [{} for i in range(times)]
    for i in range(times):
        for j in range(num_images):
            # print 'Calculating area for time %d and slice %d...' % (i, j)
            img = images[j][i]
            in_ = np.expand_dims(img, axis=0)
            in_ -= np.array([MEAN_VALUE])
            net.blobs['data'].reshape(1, *in_.shape)
            net.blobs['data'].data[...] = in_
            net.forward()
            prob = net.blobs['prob'].data
            obj = prob[0][1]
            preds = np.where(obj > THRESH, 1, 0)
            all_masks[i][j] = preds
            all_areas[i][j] = np.count_nonzero(preds)

    return all_masks, all_areas

def calc_total_volume(areas, area_multiplier, dist):
    slices = np.array(sorted(areas.keys()))
    modified = [areas[i] * area_multiplier for i in slices]
    vol = 0
    for i in slices[:-1]:
        a, b = modified[i], modified[i+1]
        subvol = (dist/3.0) * (a + np.sqrt(a*b) + b)
        vol += subvol / 1000.0  # conversion to mL
    return vol

def segment_dataset(dataset):
    # shape: num slices, num snapshots, rows, columns
    print 'Calculating areas...'
    all_masks, all_areas = calc_all_areas(dataset.images)
    print 'Calculating volumes...'
    area_totals = [calc_total_volume(a, dataset.area_multiplier, dataset.dist)
                   for a in all_areas]
    print 'Calculating EF...'
    edv = max(area_totals)
    esv = min(area_totals)
    ef = (edv - esv) / edv
    print 'Done, EF is {:0.4f}'.format(ef)

    dataset.edv = edv
    dataset.esv = esv
    dataset.ef = ef



'''Making prediction on the DSB training data'''
# Actually executing predictions
# We capture all standard output from IPython so it does not flood the interface.
with io.capture_output() as captured:
    # edit this so it matches where you download the DSB data
    DATA_PATH = './Data/Competition_data/'

    caffe.set_mode_cpu()
    net = caffe.Net('fcn_deploy.prototxt', './model_logs/fcn_iter_15000.caffemodel', caffe.TEST)

    train_dir = os.path.join(DATA_PATH, 'train')
    studies = next(os.walk(train_dir))[1]

    labels = np.loadtxt(os.path.join(DATA_PATH, 'train.csv'), delimiter=',',
                        skiprows=1)

    label_map = {}
    for l in labels:
        label_map[l[0]] = (l[2], l[1])

    if os.path.exists('output'):
        shutil.rmtree('output')
    os.mkdir('output')

    accuracy_csv = open('accuracy.csv', 'w')

    for s in studies:
        dset = Dataset(os.path.join(train_dir, s), s)
        print 'Processing dataset %s...' % dset.name
        try:
            dset.load()
            segment_dataset(dset)
            (edv, esv) = label_map[int(dset.name)]
            accuracy_csv.write('%s,%f,%f,%f,%f\n' %
                               (dset.name, edv, esv, dset.edv, dset.esv))
        except Exception as e:
            print '***ERROR***: Exception %s thrown by dataset %s' % (str(e), dset.name)

    accuracy_csv.close()

# We redirect the captured stdout to a log file on disk.
# This log file is very useful in identifying potential dataset irregularities that throw errors/exceptions in the code.
with open('logs.txt', 'w') as f:
    f.write(captured.stdout)






'''Making prediction on the DSB validation data'''
# Actually executing predictions
# We capture all standard output from IPython so it does not flood the interface.
# if __name__== "__main__":
# with io.capture_output() as captured:
    # edit this so it matches where you download the DSB data
    DATA_PATH = './Data/Competition_data/'

    caffe.set_mode_cpu()
    net = caffe.Net('fcn_deploy.prototxt', './model_logs/fcn_iter_15000.caffemodel', caffe.TEST)

    train_dir = os.path.join(DATA_PATH, 'validate')
    studies = next(os.walk(train_dir))[1]

    # labels = np.loadtxt(os.path.join(DATA_PATH, 'train.csv'), delimiter=',',
    #                     skiprows=1)

    # label_map = {}
    # for l in labels:
    #     label_map[l[0]] = (l[2], l[1])

    # if os.path.exists('output'):
    #     shutil.rmtree('output')
    # os.mkdir('output')

    accuracy_test_csv = open('./output/accuracy_test.csv', 'w')

    for s in studies:
        dset = Dataset(os.path.join(train_dir, s), s)
        print 'Processing dataset %s...' % dset.name
        try:
            dset.load()
            segment_dataset(dset)
            # (edv, esv) = label_map[int(dset.name)]
            accuracy_csv.write('%s,%f,%f\n' %
                               (dset.name, dset.edv, dset.esv))
        except Exception as e:
            print '***ERROR***: Exception %s thrown by dataset %s' % (str(e), dset.name)

    accuracy_csv.close()

   # We redirect the captured stdout to a log file on disk.
   # This log file is very useful in identifying potential dataset irregularities that throw errors/exceptions in the code.
with open('logs_test.txt', 'w') as f:
    f.write(captured.stdout)







'''
===== End of Making Predicitons =====
'''



'''
===== Evaluating Prediction and making submission file =====
'''

data = np.transpose(np.loadtxt('./output/accuracy_test.csv', delimiter=',')).astype('float')
# ids, actual_edv, actual_esv, predicted_edv, predicted_esv = data
ids, predicted_edv, predicted_esv = data
# actual_ef = (actual_edv - actual_esv) / actual_edv
# actual_ef_std = np.std(actual_ef)
# actual_ef_median = np.median(actual_ef)
predicted_ef = (predicted_edv - predicted_esv) / predicted_edv # potential of dividing by zero, where there is no predicted EDV value
nan_idx = np.isnan(predicted_ef)
# actual_ef = actual_ef[~nan_idx]
predicted_ef = predicted_ef[~nan_idx]
# MAE = np.mean(np.abs(actual_ef - predicted_ef))
# RMSE = np.sqrt(np.mean((actual_ef - predicted_ef)**2))
# print 'Mean absolute error (MAE) for predicted EF: {:0.4f}'.format(MAE)
# print 'Root mean square error (RMSE) for predicted EF: {:0.4f}'.format(RMSE)
# print 'Standard deviation of actual EF: {:0.4f}'.format(actual_ef_std)
# print 'Median value of actual EF: {:0.4f}'.format(actual_ef_median)






submit_csv = open("./output/submit_train.csv", "w")
submit_csv.write("Id,")
for i in range(0, 600):
    submit_csv.write("P%d" % i)
    if i != 599:
        submit_csv.write(",")
    else:
        submit_csv.write("\n")
for idx, s in enumerate(ids):
    submit_csv.write("%d_Systole," % int(s))
    for i in range(0, 600):
        if i < predicted_esv[idx]:
            submit_csv.write("0.0")
        else:
            submit_csv.write("1.0")
        if i == 599:
            submit_csv.write("\n")
        else:
            submit_csv.write(",")
    submit_csv.write("%d_Diastole," % int(s))
    for i in range(0, 600):
        if i < predicted_edv[idx]:
            submit_csv.write("0.0")
        else:
            submit_csv.write("1.0")
        if i == 599:
            submit_csv.write("\n")
        else:
            submit_csv.write(",")
submit_csv.close()



import pandas as pd
import re

mean_edv = np.mean(predicted_edv)
mean_esv = np.mean(predicted_esv)
df = pd.read_csv("./output/submit_train.csv")

for idx, id in enumerate(df.Id):
    pair = re.match(r'(\d+)_([a-z]+)', id)
    number=  pair.group(1)
    name = pair.group(2)
    if name == "diastolic":
        df.iloc[idx,0] = number+"_"+"Diastole"
    elif name == "systolic":
        df.iloc[idx,0] = number+"_"+"Systole"
    else:
        print"wtf"


id = [re.match(r'(\d+)_[a-z]+', ID).group(1) for ID in df.Id]

submit_csv = open("./output/submit_train.csv", "w")
for s in range(501,701):
    if s not in id:
        print s
        # submit_csv.write("%d_systolic," % int(s))
        # for i in range(0, 600):
        #     if i < mean_esv:
        #         submit_csv.write("0.0")
        #     else:
        #         submit_csv.write("1.0")
        #     if i == 599:
        #         submit_csv.write("\n")
        #     else:
        #         submit_csv.write(",")
        # submit_csv.write("%d_diastolic," % int(s))
        # for i in range(0, 600):
        #     if i < mean_edv:
        #         submit_csv.write("0.0")
        #     else:
        #         submit_csv.write("1.0")
        #     if i == 599:
        #         submit_csv.write("\n")
        #     else:
        #         submit_csv.write(",")
submit_csv.close()


