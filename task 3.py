import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score



DATADIR = "train"  
CATEGORIES = ["cat", "dog"]

IMG_SIZE = 64  

data = []
labels = []

print("Loading images...")
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)  
    class_num = CATEGORIES.index(category)  
    
    for img in tqdm(os.listdir(path)[:2000]):  
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            resized_img = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append(resized_img.flatten())  
            labels.append(class_num)
        except Exception as e:
            pass


X = np.array(data)
y = np.array(labels)

print(f"Dataset loaded: {X.shape[0]} samples, each image has {X.shape[1]} features.")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print("Training SVM model...")
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=CATEGORIES))