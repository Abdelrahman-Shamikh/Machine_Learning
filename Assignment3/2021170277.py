from skimage.feature import hog
from PIL import Image
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
def read_extract_features(folder_path):
    hog_features = []
    labels = []
    for label in os.listdir(folder_path):
        class_folder = os.path.join(folder_path, label)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                path = os.path.join(class_folder, img_name)
                image = Image.open(path).convert('L')
                resized_image = image.resize((128, 64))
                features = hog(np.array(resized_image), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
                hog_features.append(features)
                labels.append(label)
    return np.array(hog_features), np.array(labels)
train_folder = "E:\\Abdelrahman\\Machine_Learning\\Assignment3\\Assignment dataset\\train"
X_train, y_train = read_extract_features(train_folder)
test_folder = "E:\\Abdelrahman\\Machine_Learning\\Assignment3\\Assignment dataset\\test"
X_test, y_test = read_extract_features(test_folder)
classifiers = {
    'linear kernel': SVC(kernel='linear', C=1),
    'LinearSVC': svm.LinearSVC(C=1, max_iter=10000),
    'RBF kernel': SVC(kernel='rbf'),
    'polynomial kernel (degree=3)': SVC(kernel='poly', degree=3, C=1)
}
for kernel, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)  
    y_pred2=clf.predict(X_train)
    # print(y_pred)
    # print(y_test)
    accuracy2= accuracy_score(y_train, y_pred2)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"train Accuracy for {kernel}: {accuracy2}")
    print(f"test Accuracy for {kernel}: {accuracy}")
