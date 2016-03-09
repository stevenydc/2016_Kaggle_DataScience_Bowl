
'''Testing the crop_resize function'''
testpath = "./Data/Competition_data/train/1/study/sax_10/IM-4562-0001.dcm"
testf = dicom.read_file(testpath)
img = crop_resize(testf.pixel_array.astype(float) / np.max(testf.pixel_array), 64)
rawimg = testf.pixel_array.astype(float)

# plt.imshow(rawimg,'gray')
# plt.imshow(img, 'gray')
# plt.show()


'''Testing the get_data function'''
testdata, result = get_data(train_frames[0], lambda x: crop_resize(x, 64))





'''Testing the write_data_csv function'''  
# This part is basically write_data_csv but with smaller training/testing size to enable CPU prototyping
imgs = []
results = []
unzipped_results = []
for frame in train_frames[:100]:
    unzipped_results.append(get_data(frame, lambda x: crop_resize(x, 64)))
unzipped_results = np.array(unzipped_results)
print unzipped_results.shape
# Each entry has a tuple of two elements. First element is the concatenated image data of 30 frames of a single slice
# second element is a list of the image file locations of the 30 frames
print unzipped_results[0].shape
zipped_results = zip(*unzipped_results)

print len(zipped_results)
# data is an array of lists, each with 122880 entries
# file_paths is an array of lists as well, each with 30 addresses
data, file_paths = zipped_results
test_reshaped_image = data[0].reshape(30,64,64).astype(int)

#check to see if reshaped image is the right format
plt.imshow(test_reshaped_image[15,:,:])

fdata = open("./train_10_img_64x64.csv", "w")
for entry in data:
    fdata.write(",".join(entry)+'\r\n')
