import cv2
from matplotlib import pyplot as plt
import numpy as np
from mnist import MNIST
from utils import extract_features, extract_y_trains,model_selection
from math import sqrt
import scipy.misc
mndata = MNIST('traindata')
images, labels = mndata.load_training()
print(len(labels))
val = int(sqrt(len(images[6])))
imgs = [np.reshape(i,[val,val]) for i in images]
x_s = [extract_features(x,4) for x in imgs]
x_train = np.array(x_s[:40000])
x_val = np.array(x_s[40000:])
y_trains = extract_y_trains(labels[:40000])
y_vals = extract_y_trains(labels[40000:])
print("train data prepared")
w0 = np.reshape([4.0 for i in range(49)],[-1,1])
thetas = [0.05*i for i in range(2,16)]
lambdas = [0.002,0.001,0.003,0.004]
with open("params_adjusted.txt","w") as file:
    params = {}
    for i in range(10):
        params = model_selection(x_train,y_trains[i],x_val,y_vals[i],w0,200,0.3,200,lambdas,thetas)
        params[str(i)] = params
    file.write(params)