import os
import cv2
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np

def extract(image):
    resized_image = cv2.resize(image, (128, 64))
    f = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), channel_axis=-1)
    return f

trainF = "train"
testF = "test"

f = []
l = []
g=[]
t=[]
for classF in os.listdir(trainF):
    class_path = os.path.join(trainF, classF)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            hog_features = extract(image)
            f.append(hog_features)
            l.append(classF)
for classF in os.listdir(testF):
    class_path = os.path.join(testF, classF)
    if os.path.isdir(class_path):
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            image = cv2.imread(image_path)
            hog_features = extract(image)
            g.append(hog_features)
            t.append(classF)

f = np.array(f)
l = np.array(l)
g=np.array(g)
t=np.array(t)
kernels = ['poly', 'linear', 'sigmoid', 'rbf']
for kernel in kernels:
    clf = svm.SVC(kernel=kernel)
    clf.fit(f, l)
    y_pred = clf.predict(g)
    y_pred2 = clf.predict(f)
    accuracy = accuracy_score(t, y_pred)
    print(f"testing accuracy with {kernel} kernel:", accuracy)
    accuracy2 = accuracy_score(l, y_pred2)
    print(f"training accuracy with {kernel} kernel:", accuracy2)