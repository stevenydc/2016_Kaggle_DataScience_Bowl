

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
        accuracy_test_csv.write('%s,%f,%f\n' %
                           (dset.name, dset.edv, dset.esv))
    except Exception as e:
        print '***ERROR***: Exception %s thrown by dataset %s' % (str(e), dset.name)

accuracy_test_csv.close()





'''
Visualizing individual predictions
'''
dset = Dataset(os.path.join(train_dir, studies[0]), studies[0])
dset.load()

all_masks, all_areas = calc_all_areas(dset.images)

print "the length of all_masks", len(all_masks), "the size of each mask,", all_masks[0]
# for
# plt.imshow(all_masks[0][0])


