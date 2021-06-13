import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
from PIL import Image
import os
import scipy.io as sio
import time


# Make sure that caffe is on the python path:
caffe_root = '../../'  # this file is expected to be in {caffe_root}/examples/hed/
import sys

sys.path.insert(0, caffe_root + 'python')

import caffe

data_root = '../../data/crack/GAPs/'
#load test data list
with open(data_root + 'test.txt') as f:
    test_lst = f.readlines
test_f = open(data_root + 'test.txt', 'r')
# test image list
im_test_list = []
#the corresponding mask image list
mask_test_list = []
for im_pairs in test_f:
    im_test_list.append(im_pairs.split(' ')[0])
    mask_test_list.append(im_pairs.split(' ')[1])

im_test_name_list=im_test_list
im_test_list = [data_root + x.strip() for x in im_test_list]
mask_test_list = [data_root + x.strip() for x in mask_test_list]

im_lst = []
#load image and substract mean value
for i in range(0, len(im_test_list)):
    im = Image.open(im_test_list[i])
    im = np.array(im, dtype=np.float32)
    print('start--------------------')
    print(im.shape)
    im = im[:, :, ::-1]
    print(im.shape)
    print('end--------------------')
    im -= np.array((127.00699, 126.66877, 126.67892))
    im_lst.append(im)

# Visualization of detection result
def plot_single_scale(scale_lst, size):
    pylab.rcParams['figure.figsize'] = size, size / 2

    plt.figure()
    for i in range(0, len(scale_lst)):
        s = plt.subplot(1, 5, i + 1)
        plt.imshow(1 - scale_lst[i], cmap=cm.Greys_r)
        image_name = 'output_' + str(i) + '.png'
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
    plt.tight_layout()





def image_compose(imag,imag_1,compose_name):    
    import PIL.Image as Image
    to_image = Image.new('RGB', (2 * 256, 1 * 128))
    
    rom_image = Image.open(imag).resize((256,128), Image.ANTIALIAS)
    rom_image_1 = Image.open(imag_1).resize((256,128), Image.ANTIALIAS)
    to_image.paste(rom_image,(0,0))
    to_image.paste(rom_image_1,(256,0))
    to_image.save(compose_name)

# Save detection results
def save_single_scale(scale_lst, name, outputpath, image_original):
    for i in range(0, len(scale_lst)):
        from PIL import Image
        image_name = name  + '.png' 
        original_image_name = name + '_original' + '.png'       
        compose_name = name + '_compose' + '.png'
        
        plt.imsave(outputpath + image_name, 1 - scale_lst[i], cmap='gray')
        plt.imsave(outputpath + original_image_name, image_original, cmap='gray')
        image_compose(outputpath + image_name, outputpath + original_image_name, outputpath + compose_name)
        sio.savemat(outputpath + name, {'predmap': 1 - scale_lst[i]})


# set GPU model and ID
caffe.set_mode_gpu()
caffe.set_device(0)
model_root = './'
testnet='test_fphb_crack.prototxt'
valresults='./'
modelList=['fphb_crack']#['train_val_fuse_fpn_crack_c1_v2_iter_4000']
#modelList=['train_val_fuse_fpn_crack_c1_fast_v2_iter_4000','train_val_fuse_fpn_crack_c1_fast_v2_iter_8000','train_val_fuse_fpn_crack_c1_fast_v2_iter_12000','train_val_fuse_fpn_crack_c1_fast_v2_iter_16000','train_val_fuse_fpn_crack_c1_fast_v2_iter_20000','train_val_fuse_fpn_crack_c1_fast_v2_iter_24000','train_val_fuse_fpn_crack_c1_fast_v2_iter_28000','train_val_fuse_fpn_crack_c1_fast_v2_iter_32000','train_val_fuse_fpn_crack_c1_fast_v2_iter_36000','train_val_fuse_fpn_crack_c1_fast_v2_iter_40000']
for model in modelList:
    #load network architecture and trained model
    net = caffe.Net(testnet, model_root +model+'.caffemodel', caffe.TEST)
    idx = 0
    outputpath=valresults+model+'/'
    time_avg=0    
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    for i,im in enumerate(im_lst):
        im_bk = im.copy()               
        im = im.transpose((2, 0, 1))
        net.blobs['data'].reshape(1, *im.shape)
        net.blobs['data'].data[...] = im
        # run net and take argmax for prediction
        start = time.time()
        net.forward()
        end = time.time()
        time_per_image=end-start
        print('time each image',time_per_image)
        time_avg=time_avg+time_per_image
        print('time per image is %f' %time_per_image)
        out1 = net.blobs['sigmoid-dsn1'].data[0][0, :, :]
        out2 = net.blobs['sigmoid-dsn2'].data[0][0, :, :]
        out3 = net.blobs['sigmoid-dsn3'].data[0][0, :, :]
        out4 = net.blobs['sigmoid-dsn4'].data[0][0, :, :]
        out5 = net.blobs['sigmoid-dsn5'].data[0][0, :, :]
        fuse = net.blobs['sigmoid-fuse'].data[0][0, :, :]
        scale_lst=[fuse]
        print i
        #get image name
        imgname=im_test_name_list[i].split('/')[1].split('.')[0]
        print imgname
        #plot_single_scale(scale_lst, 22)
        im_bk += np.array((127.00699, 126.66877, 126.67892))
        im_bk = im_bk[:,:,0]
        save_single_scale(scale_lst, imgname, outputpath, im_bk)

    print('average time is %f' %(time_avg/len(im_lst)))
