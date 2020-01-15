from __future__ import division
import numpy as np
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib as mpl
import matplotlib.pyplot as plt
from bonnerlib2 import dfContour
import pickle
import time

#Question 1(a)

def gen_data(mu0, mu1, cov0, cov1, N0, N1):
    """
    Generates the data points and target values for two clusters, using 
    the means, covariances and numbers of points given in the arguments. 
    
    :param mu0: mean of cluster 0.
    :param mu1: mean of cluster 1.
    :param cov0: covariance of cluster 0.
    :param cov1: covariance of cluster 1.
    :param N0: number of points in cluster 0.
    :param N1: number of points in cluster 1.
    :return: X: data points
             t: target values
    """
    # Generate the data for a cluster
    cov_mat0 = [[1, cov0], [cov0, 1]]
    cov_mat1 = [[1, cov1], [cov1, 1]]
    cluster0 = np.random.multivariate_normal(mu0, cov_mat0, N0)
    cluster1 = np.random.multivariate_normal(mu1, cov_mat1, N1)

    # Combine the two clusters to create X and t
    X = np.concatenate((cluster0, cluster1))
    t_zeros = np.zeros(cluster0.shape[0])
    t_ones = np.ones(cluster1.shape[0])
    t = np.concatenate((t_zeros, t_ones))
    
    # Distribute the data randomly in the two arrays
    X, t = sk.utils.shuffle(X, t)

    return X, t

def plot_data(X, t):
    """
    Helper function to plot two clusters using data points in X
    and target values in t.
    
    :param X: array of data points.
    :param t: array of target values that corresponds to each data point.
    :return: null
    """
    colours = ['red', 'blue']
    plt.scatter(X[:, 0], X[:, 1],c=t, cmap=mpl.colors.ListedColormap(colours), s=2)
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)

def draw_prob_contour(X, t, w, w0, prob, colour):
    """
    Helper function to draw a decision boundary for two clusters,
    with the probability and colour of the line given.
    
    :param X: an array of data point.
    :param t: an array of target values corresponding to each data point.
    :param w: the weights of the model, returned by a classifier.
    :param w0: the bias term of the model, returned by a classifier.
    :param prob: the probability at which to draw the decision
    :param colour: the colour we want the decision boundary on the plot.
    :return: null
    """
    z = (np.log(1/prob-1))
    x_lims = np.array((-3.0, 6.0))
    y = -(x_lims * w[0,0] + w0[0]+z)/w[0,1]
    plt.plot(x_lims, y, c=colour)
    plt.xlim(-3, 6)
    plt.ylim(-3, 6)
    
def get_tp_fp_fn(Xtest, Ttest, prob):
    """
    """
    z_log_reg = np.matmul(Xtest, w.T) + w0
    p_log_reg = 1/(1+np.exp(-(z_log_reg)))
    class_array = p_log_reg > prob
    class_array = class_array.astype(int)
    class_array = class_array.flatten()
    Ttest = Ttest.flatten()
    
    #total_differences = np.count_nonzero(diff==0)
    true_pos = np.sum((Ttest == 1) & (class_array == 1))
    false_pos = np.sum((class_array == 1) & (Ttest == 0))
    false_neg = np.sum((class_array == 0) & (Ttest == 1))
    
    return true_pos, false_pos, false_neg
# Question 1(b)
X, t = gen_data([1, 1], [2, 2], 0, -0.9, 10000, 5000)

# Question 1(c)
plt.figure(1)
plot_data(X, t)
plt.title("Question 1(c): sample cluster data")

# Question 2(a)
X_train, t_train = gen_data([1, 1], [2, 2], 0, -0.9, 1000, 500)

# Question 2(b)
# Logistic Regression on the new X, t
print "\nQuestion 2(b)"
print "\n---------------"
log_reg = LogisticRegression()
log_reg.fit(X_train, t_train)
w = log_reg.coef_
w0 = log_reg.intercept_
print "Bias term w0: ", w0
print "Weight vector w: ", w

# Question 2(c)
# using score method
print "\nQuestion 2(c)"
print "\n---------------"
accuracy1 = log_reg.score(X_train, t_train)
# manually
z_log_reg = np.matmul(X_train, w.T) + w0
p_log_reg = 1/(1+np.exp(-(z_log_reg)))
class_array = p_log_reg > 0.5
class_array = class_array.astype(int)
class_array = class_array.flatten()
diff = np.equal(class_array, t_train)
diff = diff.astype(int)
#total_differences = np.count_nonzero(diff==0)
total_differences = np.where( diff == 0)[0].shape[0]
accuracy2 = (t_train.shape[0] - total_differences)/t_train.shape[0]

print "accuracy1: ", accuracy1
print "accuracy2: ", accuracy2
print "difference: ", accuracy1 - accuracy2

# Question 2(d)
plt.figure(2)
plot_data(X_train, t_train)
x_lims = np.array((-3.0, 6.0))
# Draw the decision boundary
y = -(x_lims * w[0,0] + w0[0])/w[0,1]
plt.plot(x_lims, y, c="black")
plt.title("Question 2(d): training data and decision boundary")

# Question 2(e)
plt.figure(3)
plot_data(X_train,t_train)
draw_prob_contour(X_train,t_train, w, w0, 0.05, "red")
draw_prob_contour(X_train,t_train, w, w0, 0.5, "black")
draw_prob_contour(X_train,t_train, w, w0, 0.6, "blue")
plt.title("Question 2(e): three contours")

# Question 2(f)
X_test, t_test = gen_data([1, 1], [2, 2], 0, -0.9, 10000, 5000)

# Question 2(g)
print "\nQuestion 2(g)"
print "\n---------------"
#total_differences = np.count_nonzero(diff==0)
true_pos_red, false_pos_red, false_neg_red = get_tp_fp_fn(X_test, t_test, 0.05)
true_pos_black, false_pos_black, false_neg_black = get_tp_fp_fn(X_test, t_test, 0.5)
true_pos_blue, false_pos_blue, false_neg_blue = get_tp_fp_fn(X_test, t_test, 0.6)

precision_red = np.float(true_pos_red) / np.float(np.sum(true_pos_red + false_pos_red))
recall_red = np.float(true_pos_red) / np.float(np.sum(true_pos_red + false_neg_red))
precision_black = np.float(true_pos_black) / np.float(np.sum(true_pos_black + false_pos_black))
recall_black = np.float(true_pos_black) / np.float(np.sum(true_pos_black + false_neg_black))
precision_blue = np.float(true_pos_blue) / np.float(np.sum(true_pos_blue + false_pos_blue))
recall_blue = np.float(true_pos_blue) / np.float(np.sum(true_pos_blue + false_neg_blue))
print "precision for P(C=1|x) = 0.05: ", precision_red
print "recall for P(C=1|x) = 0.05: ", recall_red
print "precision for P(C=1|x) = 0.5: ", precision_black
print "recall for P(C=1|x) = 0.5: ", recall_black
print "precision for P(C=1|x) = 0.6: ", precision_blue
print "recall for P(C=1|x) = 0.6: ", recall_blue

# Question 4(a)
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, t_train)
accuracy4a = qda.score(X_test, t_test)
print "accuracy 4(a): ", accuracy4a
plt.figure(4)
plot_data(X_train, t_train)
dfContour(qda)
plt.title("Question 4(a): Decision boundary and contours")

# Question 4(c) 
X_train_new, t_train_new = gen_data([1, 1], [2, 2], 0, 0.9, 1000, 500)
qda.fit(X_train_new, t_train_new)
X_test_new, t_test_new = gen_data([1, 1], [2, 2], 0, 0.9, 10000, 5000)
accuracy_qda = qda.score(X_test_new, t_test_new)
print "accuracy 4(c): ", accuracy_qda
plt.figure(5)
plot_data(X_train_new, t_train_new)
dfContour(qda)
plt.title("Question 4(c): Decision boundary and contours")

# Question 4(d)
X_train_new2, t_train_new2 = gen_data([1, 1], [2, 2], 0, 0.9, 1000, 5000)
qda.fit(X_train_new2, t_train_new2)
X_test_new2, t_test_new2 = gen_data([1, 1], [2, 2], 0, 0.9, 10000, 50000)
accuracy_qda = qda.score(X_test_new2, t_test_new2)
print "accuracy 4(d): ", accuracy_qda
plt.figure(6)
plot_data(X_train_new2, t_train_new2)
dfContour(qda)
plt.title("Question 4(d): Decision boundary and contours")

# Question 4(e)
def myGDA(Xtrain, Ttrain, Xtest, Ttest):
    Ttrain_zeros_inds = np.where(Ttrain == 0)
    Ttrain_ones_inds = np.where(Ttrain == 1)
    phi1 = len(Ttrain_ones_inds[0])/len(Ttrain)
    phi0 = 1 - phi1
    
    mu0_0 = np.sum(np.take(Xtrain[:, 0], Ttrain_zeros_inds))/len(Ttrain_zeros_inds[0])
    mu0_1 = np.sum(np.take(Xtrain[:, 1], Ttrain_zeros_inds))/len(Ttrain_zeros_inds[0])
    mu0 = [[mu0_0], [mu0_1]]
    mu0 = np.array(mu0)
    mu1_0 = np.sum(np.take(Xtrain[:, 0], Ttrain_ones_inds))/len(Ttrain_ones_inds[0])
    mu1_1 = np.sum(np.take(Xtrain[:, 1], Ttrain_ones_inds))/len(Ttrain_ones_inds[0])
    mu1 = [[mu1_0], [mu1_1]]
    mu1 = np.array(mu1)
    X_0 = Xtrain[Ttrain == 0]
    
    Xtrain_minus_mu0_0 = np.abs(Xtrain[:,0] - mu0[0])
    Xtrain_minus_mu0_1 = np.abs(Xtrain[:,1] - mu0[1])
    cov_Xtrain_minus_mu0 = [[np.sum(Xtrain_minus_mu0_0)/len(Ttrain)], [np.sum(Xtrain_minus_mu0_1)/len(Ttrain)]]
    cov_Xtrain_minus_mu0 = np.array(cov_Xtrain_minus_mu0)
    cov0 = np.matmul(((cov_Xtrain_minus_mu0)), ((cov_Xtrain_minus_mu0).T))
    Xtrain_minus_mu1_0 = np.abs(Xtrain[:,0] - mu1[0])
    Xtrain_minus_mu1_1 = np.abs(Xtrain[:,1] - mu1[1])
    cov_Xtrain_minus_mu1 = [[np.sum(Xtrain_minus_mu1_0)/len(Ttrain)], [np.sum(Xtrain_minus_mu1_1)/len(Ttrain)]]
    cov_Xtrain_minus_mu1 = np.array(cov_Xtrain_minus_mu1)
    cov1 = np.matmul(((cov_Xtrain_minus_mu1)), ((cov_Xtrain_minus_mu1).T))
    
    Xtrain_minus_mu0 = [Xtrain_minus_mu0_0, Xtrain_minus_mu0_1]
    Xtrain_minus_mu0 = np.array(Xtrain_minus_mu0)
    Xtrain_minus_mu1 = [Xtrain_minus_mu1_0, Xtrain_minus_mu1_1]
    Xtrain_minus_mu1 = np.array(Xtrain_minus_mu1)

    prob0_num = np.exp(-(np.matmul(np.matmul((Xtrain_minus_mu0).T, np.linalg.inv(cov0)), Xtrain_minus_mu0))/2)
    prob0 = prob0_num/np.sqrt(np.linalg.det(cov0))
    
print "\nQuestion 4(e)"
print "\n----------------"
myGDA(X_train, t_train, X_test, t_test)
print "accuracy4a: ", accuracy4a

with open('mnist.pickle','rb') as f:
    Xtrain,Ttrain,Xtest,Ttest = pickle.load(f)
    

# Question 5(a)
random_images = Xtrain[np.random.choice(Xtrain.shape[0], size=25, replace=False), :]

plt.figure(7)
plt.title("Question 5(a): 25 random MNIST images.")
plt.axis('off')
# Generate the 5X5 grid
for i in range(1,25):
    this_image = random_images[i].reshape(28,28)
    plt.subplot(5, 5, i)
    plt.imshow(this_image,cmap="Greys", interpolation="nearest")
    plt.axis('off')

# Question 5(b)
print "\nQuestion 5(b): "
print "\n------------------"
clf = QuadraticDiscriminantAnalysis()
start_time = time.time()
clf.fit(Xtrain, Ttrain)
end_time = time.time()
total_time = end_time - start_time
print "5(b): time to fit: ", total_time
accuracy5btrain = clf.score(Xtrain, Ttrain)
accuracy5btest = clf.score(Xtest, Ttest)
print "accuracy5btrain: ", accuracy5btrain
print "accuracy5btest: ", accuracy5btest
    
# Question 5(c)
clf = GaussianNB()
start_time = time.time()
clf.fit(Xtrain, Ttrain)
end_time = time.time()
total_time = end_time - start_time
print "5(c): time to fit: ", total_time
accuracy5ctrain = clf.score(Xtrain, Ttrain)
accuracy5ctest = clf.score(Xtest, Ttest)
print "accuracy5ctrain: ", accuracy5ctrain
print "accuracy5ctest: ", accuracy5ctest

# Question 5(d): 
sigma = 0.1
noise = sigma*np.random.normal(size=np.shape(Xtrain))
Xtrain = Xtrain + noise
random_images = Xtrain[np.random.choice(Xtrain.shape[0], size=25, replace=False), :]

plt.figure(8)
plt.title("Question 5(d): 25 random MNIST images with noise")
# Generate the 5X5 grid
for i in range(1,25):
    this_image = random_images[i].reshape(28,28)
    plt.subplot(5, 5, i)
    plt.imshow(this_image,cmap="Greys", interpolation="nearest")
    plt.axis('off')
    
# Question 5(e)
clf = QuadraticDiscriminantAnalysis()
start_time = time.time()
clf.fit(Xtrain, Ttrain)
end_time = time.time()
total_time = end_time - start_time
print "5(e): time to fit: ", total_time
accuracy5btrain = clf.score(Xtrain, Ttrain)
accuracy5btest = clf.score(Xtest, Ttest)
print "accuracy5etrain: ", accuracy5btrain
print "accuracy5etest: ", accuracy5btest

# Question 5(f)
Xtrain_red = Xtrain[0:5999,:]
Ttrain_red = Ttrain[0:5999]
clf = QuadraticDiscriminantAnalysis()
start_time = time.time()
clf.fit(Xtrain, Ttrain)
end_time = time.time()
total_time = end_time - start_time
print "5(f): time to fit: ", total_time
accuracy5btrain = clf.score(Xtrain_red, Ttrain_red)
accuracy5btest = clf.score(Xtest, Ttest)
print "accuracy5ftrain: ", accuracy5btrain
print "accuracy5ftest: ", accuracy5btest

clf = GaussianNB()
start_time = time.time()
clf.fit(Xtrain_red, Ttrain_red)
end_time = time.time()
total_time = end_time - start_time
print "5(e): time to fit: ", total_time
accuracy5ctrain = clf.score(Xtrain_red, Ttrain_red)
accuracy5ctest = clf.score(Xtest, Ttest)
print "accuracy5ftrain: ", accuracy5ctrain
print "accuracy5ftest: ", accuracy5ctest

# Question 5(g)
print "clf.theta_: ", clf.theta_
print "clf.theta_.shape: ", clf.theta_.shape
plt.figure(9)
plt.title("Question 5(g): means for each digit class.")
# Generate the 5X5 grid
for i in range(1,10):
    this_image = clf.theta_[i].reshape(28,28)
    plt.subplot(3, 4, i)
    plt.imshow(this_image,cmap="Greys", interpolation="nearest")
    plt.axis('off')


