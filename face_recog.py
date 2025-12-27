import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.svm import SVC

##Helper functions. Use when needed. 
def show_orignal_images(pixels):
    fig, axes = plt.subplots(6, 10, figsize=(11, 7),subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.array(pixels)[i].reshape(64, 64), cmap='gray')
    plt.show()

def show_eigenfaces(pca):
    fig, axes = plt.subplots(3, 8, figsize=(9, 4),subplot_kw={'xticks':[], 'yticks':[]})
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(64, 64), cmap='gray')
        ax.set_title("PC " + str(i+1))
    plt.show()

## Step 1: Read dataset-----------
df = pd.read_csv("face_data.csv")

# Get labels
labels = df["target"]
pixels = df.drop(["target"], axis=1)


## Split dataset into trainign and testing------------
x_train, x_test, y_train, y_test = train_test_split(pixels, labels)

## PCA -----------

# TEST OUT ALL FIRST
# pca = PCA(n_components=200).fit(x_train)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))          # cumsum = cumulative sum. shows us how much variance we are capturing as we add more n_components
# plt.show()

# Upon testing, 135 cases / n_components needed for 96.5% of the data.
pca = PCA(n_components=135).fit(x_train)

## PROJECT/TRANFORM training data onto pca------------
x_train_pca = pca.transform(x_train)

## SET CLASSIFIER and fit training data
clf = SVC(kernel='rbf',C=1000,gamma=0.001)
clf = clf.fit(x_train_pca, y_train)

## Step 6: Perform testing and get classification report
print("Predicting people's names on the test set")
t0 = time()
Xtest_pca = pca.transform(x_test)
y_pred = clf.predict(Xtest_pca)
print("done in %0.3fs" % (time() - t0))
print(classification_report(y_test, y_pred))