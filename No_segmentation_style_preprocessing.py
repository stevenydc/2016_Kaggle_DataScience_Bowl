__author__ = 'stevenydc'
'''
===== Loading Libraries =====
'''
import dicom, lmdb, re, sys
import os, fnmatch, shutil, subprocess
import csv, random, scipy.misc
from sklearn.externals.joblib import Parallel, delayed
from skimage import transform
import matplotlib.pyplot as plt


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
===== Loading and Formatting data sets =====
'''
def get_frames(root_path):
    """Get path to all the frame in view SAX and contain complete frames"""
    ret = []
    for root, _, files in os.walk(root_path):
        # root=root.replace('\\','/')
        files=[s for s in files if ".dcm" in s]
        if len(files) == 0 or root.find("sax") == -1:
            continue
        prefix = files[0].rsplit('-', 1)[0]
        fileset = set(files)
        expected = ["%s-%04d.dcm" % (prefix, i + 1) for i in range(30)]
        if all(x in fileset for x in expected):
            ret.append([root + "/" + x for x in expected])
    print "Finished getting all frames!"
    # sort for reproduciblity
    return sorted(ret, key = lambda x: x[0])


def get_label_map(fname):
    labelmap = {}
    fi = open(fname)
    fi.readline()
    for line in fi:
        arr = line.split(',')
        labelmap[int(arr[0])] = line
    print "Finished getting all label map (first entry in line: line)"
    return labelmap



def write_label_csv(fname, frames, label_map):
    # This function creates the label csv file corresponding each *slice*.
    # Notice that there are 30 time points in each slice, and each study can have multiple slices (p)
    # Each study has a unique label... so the resulting file contains multiple identical lines for each study
    fo = open(fname, "w")
    for lst in frames:
        index = int(lst[0].split("/")[4])
        if label_map != None:
            fo.write(label_map[index])
        else:
            fo.write("%d,0,0\n" % index)
    print "Finished creating label csv."
    fo.close()




def get_data(lst,preproc):
    # lst is a set of images of one single slice at different time points
    data = []
    result = []
    # print ""
    for path in lst:
        f = dicom.read_file(path)
        # Be careful that we have to divide all entries by max value
        # other wise return value of preproc may be out of range
        img = preproc(f.pixel_array.astype(float) / np.max(f.pixel_array))
        dst_path = path.rsplit(".", 1)[0] + ".64x64.jpg"
        scipy.misc.imsave(dst_path, img)
        result.append(dst_path)
        data.append(img)
    data = np.array(data, dtype=np.uint8)
    print "initial data length", data.shape
    data = data.reshape(data.size)
    print "data.size = ", data.size
    data = np.array(data,dtype=np.str_)
    print "data has now been transformed to 3 character long string"
    print "new size,", data.size
    data = data.reshape(data.size)
    print "data is transformed to new shape", data.shape
    return [data,result]



def write_data_csv(fname, frames, preproc):
    """Write data to csv file"""
    fdata = open(fname, "w")
    # The following line is very interesting
    #
    dr = Parallel()(delayed(get_data)(lst,preproc) for lst in frames)
    #zip(*dr) turns the original dr, which is a long list of tuples of length 2,
    #in to two long lists of single elements.
    data,result = zip(*dr)
    for entry in data:
        fdata.write(','.join(entry)+'\r\n')
    print("All finished, %d slices in total" % len(data))
    fdata.close()
    result = np.ravel(result)
    return result


def crop_resize(img, size):
    """crop center and resize"""
    if img.shape[0] < img.shape[1]:
        img = img.T
    # we crop image from center to make it square first.
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    # resize to 64, 64
    resized_img = transform.resize(crop_img, (size, size))
    resized_img *= 255

    # print max(resized_img[0])
    # plt.imshow(resized_img, 'gray')
    # plt.show()
    return resized_img.astype("uint8") #TODO: this uint8 thing changes the qualify of the pic
                                      #Consinder making it float instead?



def local_split(train_index):
    random.seed(0)
    train_index = set(train_index)
    all_index = sorted(train_index)
    num_test = int(len(all_index) / 3)
    random.shuffle(all_index)
    train_set = set(all_index[num_test:])
    test_set = set(all_index[:num_test])
    return train_set, test_set

def split_csv(src_csv, split_to_train, train_csv, test_csv):
    ftrain = open(train_csv, "w")
    ftest = open(test_csv, "w")
    cnt = 0
    for l in open(src_csv):
        if split_to_train[cnt]:
            ftrain.write(l)
        else:
            ftest.write(l)
        cnt = cnt + 1
    ftrain.close()
    ftest.close()




# with io.capture_output() as captured:
TRAIN_PATH      = "./Data/Competition_data/train"
VALIDATE_PATH   = "./Data/Competition_data/validate"
random.seed(10)
train_frames = get_frames(TRAIN_PATH)
random.shuffle(train_frames)
validate_frames = get_frames(VALIDATE_PATH)
label_map = get_label_map("./Data/Competition_data/train.csv")

# Following two lines create csv file for the label, which matches the total number of slices
# Should only be run once. 
write_label_csv("./Data/Competition_data/train-label.csv", train_frames, label_map)
write_label_csv("./Data/Competition_data/validate-label.csv", validate_frames, None)

train_lst = write_data_csv("./train-64x64-data.csv", train_frames, lambda x: crop_resize(x, 64))
valid_lst = write_data_csv("./validate-64x64-data.csv", validate_frames, lambda x: crop_resize(x, 64))



def write_data_lmdb(fname, frames, preproc):
    # Get all data from original dicom dataset
    dr = Parallel()(delayed(get_data)(lst,preproc) for lst in frames)
    data,result = zip(*dr)
    #generate my imdb
    db_imgs = lmdb.open(fname)
    counter_img = 0
    with db_imgs.begin(write=True) as txn_img:
        for entry in data:
            datum = caffe.io.array_to_datum(entry.reshape(30,64,64).astype(int))
            txn_img.put("{:0>10d}".format(counter_img),datum.SerializeToString())
            counter_img += 1
    print "processed {:d} images".format(counter_img)
    lmdb.close()
    return 


def csv_to_imdb(csv_filename, lmdbname):
    # generating testing training imdb from csv file:
    db_imgs_fromfile = lmdb.open(lmdbname)
    counter_img=0
    with db_imgs_fromfile.begin(write=True) as txn_img:
        with open(csv_filename, "rb") as csvfile:
            datareader = csv.reader(csvfile)
            for row in datareader:
                # print len(row)
                # temp_img = np.array(row).reshape(30,64,64).astype(int)[0,:,:]
                # plt.imshow(temp_img)
                datum = caffe.io.array_to_datum(np.array(row).reshape(30,64,64).astype(int))
                txn_img.put("{:0>10d}".format(counter_img),datum.SerializeToString())
                counter_img += 1
    lmdb.close()
    return


def creat_prototype_datasets(proto_name, train_frames, train_size, valid_size):
    pass
#generating my prototype label lmdb
#First we need to generate our label csv
write_label_csv("./prototype_train_label.csv",train_frames[:100], label_map)
#Now we read from fril and generate lmdb:
db_label_fromfile = lmdb.open("")
counter_label=0
with db_label_fromfile.begin(write=True) as txn_img3:
    with open("./train-label-10-img-64x64.csv", "rb") as csvfile:
        datareader = csv.reader(csvfile)
        for row in datareader:
            print len(row), "\t:", row

            # temp_img = np.array(row).reshape(30,64,64).astype(int)[0,:,:]
            # plt.imshow(temp_img)
            # datum = caffe.io.array_to_datum(np.array(row).reshape(30,64,64).astype(int))
            # txn_img3.put("{:0>10d}".format(counter_img),datum.SerializeToString())
            # counter_img += 1


# Generate local train/test split, which you could use to tune your model locally.
train_index = np.loadtxt("./train-label.csv", delimiter=",")[:,0].astype("int")
train_set, test_set = local_split(train_index)
split_to_train = [x in train_set for x in train_index]
split_csv("./train-label.csv", split_to_train, "./local_train-label.csv", "./local_test-label.csv")
split_csv("./train-64x64-data.csv", split_to_train, "./local_train-64x64-data.csv", "./local_test-64x64-data.csv")



# with open('logs.txt', 'w') as f:
#     f.write(captured.stdout)





'''
Transforming label to 600 CDF, and store in csv?
'''
#TODO: consider directly saving to a lmdb or something
def encode_label(label_data):
    """Run encoding to encode the label into the CDF target.
    """
    systole = label_data[:, 1]
    diastole = label_data[:, 2]
    systole_encode = np.array([
            (x < np.arange(600)) for x in systole
        ], dtype=np.uint8)
    diastole_encode = np.array([
            (x < np.arange(600)) for x in diastole
        ], dtype=np.uint8)
    return systole_encode, diastole_encode


#Originally encode_csv, modified to lmdb to suit Caffe
#Check out the original encode_csv from MXnet Kaggle Data Science Bowl Github
def encode_lmdb(label_csv, systole_lmdb, diastole_lmdb):
    # Function takes in the processed label file (through write_label_csv) and encode them
    # to CDF, with systole and diastole separated. 
    # systole_encode and iastole_encode are two 2D arrays, with 10 entries, and each entry is a row of 600 entries
    systole_encode, diastole_encode = encode_label(np.loadtxt(label_csv, delimiter=","))
    systole_count = 0
    diastole_count = 0
    print "There are {0} many slices of systole data to encode.".format(len(systole_encode))
    print "There are {0} many slices of systole data to encode.".format(len(diastole_encode))
    encoded_label_systole_lmdb = lmdb.open(systole_lmdb, map_size =1e12)
    encoded_label_diastole_lmdb = lmdb.open(diastole_lmdb, map_size =1e12)

    with db_10img_encoded_label_systole.begin(write=True) as txn_img:
        for label in systole_encode:
            # print np.expand_dims(np.expand_dims(label, axis=1), axis=1).shape
            datum = caffe.io.array_to_datum(np.expand_dims(np.expand_dims(label, axis=1), axis=1))
            txn_img.put("{:0>10d}".format(systole_count),datum.SerializeToString())
            systole_count+=1

    with db_10img_encoded_label_diastole.begin(write=True) as txn_img:
        for label in diastole_encode:
            # print np.expand_dims(np.expand_dims(label, axis=1), axis=1).shape
            datum = caffe.io.array_to_datum(np.expand_dims(np.expand_dims(label, axis=1), axis=1))
            txn_img.put("{:0>10d}".format(diastole_count),datum.SerializeToString())
            diastole_count+=1

    # np.savetxt(systole_csv, systole_encode, delimiter=",", fmt="%g")
    # np.savetxt(diastole_csv, diastole_encode, delimiter=",", fmt="%g")

# Write encoded label into the target csv
# We use CSV so that not all data need to sit into memory
# You can also use inmemory numpy array if your machine is large enough
encode_csv("./train-label-10-img-64x64.csv", "./train-64x64-10img-systole.csv", "./train-64x64-10img-diastole.csv")

'''
testing the encoding part of labels
'''
systole_encode, diastole_encode = encode_label(np.loadtxt("./train-label-10-img-64x64.csv", delimiter=","))
print len(systole_encode[0])

systole_csv = "db_10img_encoded_label_systole"
diastole_csv= "db_10img_encoded_label_diastole"


