from mnist import MNIST
from utils import extract_features, extract_y_trains, prediction, f_measure
from math import sqrt
import numpy as np
import re
import matplotlib.pyplot as plt

params_adj = {}
params = {}
with open("params_adjusted.txt", "r") as f:
    lines = f.readlines()
    for i in range(10):
        params_adj[str(i)] = eval(lines[i])
with open("params.txt", "r") as f:
    lines = f.readlines()
    for i in range(10):
        params[str(i)] = eval(lines[i])

mndata = MNIST('testdata')
images, labels = mndata.load_testing()
val = int(sqrt(len(images[6])))
imgs = [np.reshape(i, [val, val]) for i in images]
x_s_7 = np.array([extract_features(x, 7) for x in imgs])
x_s_4 = np.array([extract_features(x, 4) for x in imgs])
y_trains = extract_y_trains(labels)
labels_chart = ["Tsh/top", "Trous", "Pullov", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle"]
train_results = []
train_results_adj = []
test_results = []
test_results_adj = []
all_w = []
all_thetas = []
all_w_adj = []
all_thetas_adj = []
for i in range(10):
    w = np.array(params[str(i)][2])
    all_w.append(w)
    w_adj = np.array(params_adj[str(i)][2])
    all_w_adj.append(w_adj)
    theta = params[str(i)][1]
    all_thetas.append(theta)
    theta_adj = params_adj[str(i)][1]
    all_thetas_adj.append(theta_adj)
    train_results.append(np.mean(params[str(i)][-1]))
    train_results_adj.append(np.mean(params_adj[str(i)][-1]))
    test_results.append(f_measure(y_trains[i], prediction(x_s_7, w, theta)))
    test_results_adj.append(f_measure(y_trains[i], prediction(x_s_4, w_adj, theta_adj)))
plt.style.use('ggplot')
lab_pos = [i*2 for i,_ in enumerate(labels_chart)]
print("weaker results: ", test_results)
mean_res = np.mean(test_results)
print("mean: ", mean_res)
mean_results = [mean_res for i in range(10)]
mean_res_adj = np.mean(test_results_adj)
print("mean: ", mean_res_adj)
mean_results_adj = [mean_res_adj for i in range(10)]
print("wyniki z poprawkami: ", test_results_adj)
plt.bar(lab_pos,test_results_adj,alpha=0.7,label="stronger model results",color = "green")
plt.bar(lab_pos,mean_res_adj,alpha=0.3,label="mean stronger model result", color = "yellow")
plt.bar(lab_pos,test_results,alpha=0.7,label="weaker model results",color="red")
plt.bar(lab_pos,mean_res,alpha=0.3,label="mean weaker model result", color = "blue")
plt.xlabel("class")
plt.ylabel("efficiency")
plt.title("Models' efficiencies")
plt.xticks(lab_pos,labels_chart)
plt.legend(loc='lower right')
plt.savefig("Results.png")
plt.show()