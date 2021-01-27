# mnist-fashion-pnod4
mnist-fasion picture classification using logistic regression
## Introduction
Included code allows for solving the problem of classification 28x28 pixel fashion icons within 10 categories:
- T-shirt/top,
- Trousers,
- Pullover,
- Dress,
- Coat,
- Sandal,
- Shirt,
- Sneaker,
- Bag,
- Ankle
## Methods:
The code presents two methods for solving of the problem. Both of them use logistic regression with one vs all approach as classiffication algorithm with lambda2 regularisation and Stochastic gradient descent.


![My image](/img/mfe.png)


Features of images are extracted by division of the image into smaller squares and taking mean of the values within the square. (as seen above)
- First model (further refferanced as "Weaker") performs feature extraction by dividing original image into 16 7x7 squares. Then extracted features are divided into training set and validation set in 2/3 proportion. Then model selection is performed with regularisation parameter - lambda and likelihood treshold - theta taken as hiperparameteres. Stochastic gradient descent uses 0.3 as step parameter, 150 as number of epochs and 100 as minibach size. Initial  parameter vector is set for 0.5 for all values.
- Second model (further refferanced as "Stronger" or "Adjusted") performs feature extraction by dividing original image into 49 4x4 squares. Then extracted features are divided into training set and validation set in 2/3 proportion. Then model selection is performed with regularisation parameter - lambda and likelihood treshold - theta taken as hiperparameteres. Stochastic gradient descent uses 0.3 as step parameter, 200 as number of epochs and 200 as minibach size. Initial  parameter vector is set for 4.0 for all values.
- Model quality is measured usinf F_measure standard
## Results
|      |Tshirt/top| Trousers| Pullover| Dress| Coat | Sandal| Shirt| Sneaker| Bag | Ankle Boots| Mean |
|------|----------|---------|---------|------|------|-------|------|--------|-----|------------|------|
| weak |   0.743  |  0.862  |  0.528  | 0.643|0.465 | 0.646 | 0.303|  0.680 |0.578|    0.663   | 0.611|
|strong|   0.751  |  0.913  |   0.524 | 0.718| 0.615| 0.836 | 0.365|  0.767 |0.883|    0.875   | 0.724|

