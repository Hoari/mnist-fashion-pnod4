from matplotlib import pyplot as plt
import numpy as np
from mnist import MNIST
from utils import extract_features, extract_y_trains,model_selection
from math import sqrt
mndata = MNIST('traindata')
images, labels = mndata.load_training()
print(len(labels))
val = int(sqrt(len(images[6])))
imgs = [np.reshape(i,[val,val]) for i in images]
x_s = [extract_features(x) for x in imgs]
x_train = np.array(x_s[:40000])
x_val = np.array(x_s[40000:])
y_trains = extract_y_trains(labels[:40000])
y_vals = extract_y_trains(labels[40000:])
print("train data prepared")
w0 = np.reshape([0.5 for i in range(16)],[-1,1])
thetas = [0.1*i for i in range(2,8)]
lambdas = [0.2,0.001,0.003,0.0045]
with open("params.txt","w") as file:
    for i in range(10):
        print(i)
        file.write(str(i)+"\n")
        params = model_selection(x_train,y_trains[i],x_val,y_vals[i],w0,150,0.3,100,lambdas,thetas)
        print(params)
        file.write(str(params) +"\n")