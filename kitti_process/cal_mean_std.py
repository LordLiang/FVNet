import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from imageio import imread

filepath = '/data/dataset/kitti_fvnet2/projection'
training_set = glob.glob(os.path.join(filepath, 'training', 'images_512', '*.png'))
testing_set = glob.glob(os.path.join(filepath, 'testing', 'images_512', '*.png'))
num = len(training_set) + len(testing_set)
print(num)
size = num * 128 * 512

R_channel = 0
G_channel = 0
B_channel = 0

for filename in training_set:
    img = imread(filename) / 255.0
    R_channel = R_channel + np.sum(img[:,:,0])
    G_channel = G_channel + np.sum(img[:,:,1])
    B_channel = B_channel + np.sum(img[:,:,2])

for filename in testing_set:
    img = imread(filename) / 255.0
    R_channel = R_channel + np.sum(img[:,:,0])
    G_channel = G_channel + np.sum(img[:,:,1])
    B_channel = B_channel + np.sum(img[:,:,2])

R_mean = R_channel / size
G_mean = G_channel / size
B_mean = B_channel / size

R_channel = 0
G_channel = 0
B_channel = 0

for filename in training_set:
    img = imread(filename) / 255.0
    R_channel = R_channel + np.sum((img[:,:,0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:,:,1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:,:,2] - B_mean) ** 2)

for filename in testing_set:
    img = imread(filename) / 255.0
    R_channel = R_channel + np.sum((img[:,:,0] - R_mean) ** 2)
    G_channel = G_channel + np.sum((img[:,:,1] - G_mean) ** 2)
    B_channel = B_channel + np.sum((img[:,:,2] - B_mean) ** 2)

R_var = np.sqrt(R_channel / size)
G_var = np.sqrt(G_channel / size)
B_var = np.sqrt(B_channel / size)

print("%.5f, %.5f, %.5f" %(R_mean, G_mean, B_mean))
print("%.5f, %.5f, %.5f" %(R_var, G_var, B_var))
