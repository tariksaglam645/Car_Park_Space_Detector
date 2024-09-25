import os
import cv2 as cv
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score


class ImageClassifier:
    def __init__(self, train_dir, model_path):

        self.train_dir = train_dir
        self.model_path = model_path
        self.img_list = []
        self.class_list = []
        self.model = None

    def load_data(self):

        for id, class_folder in enumerate(os.listdir(self.train_dir)):
            class_path = os.path.join(self.train_dir, class_folder)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv.imread(img_path)
                if img is not None:
                    img = cv.resize(img, (15, 15), cv.INTER_AREA)
                    img = img.flatten()
                    self.img_list.append(img)
                    self.class_list.append(id)

        self.img_list = np.array(self.img_list)
        self.class_list = np.array(self.class_list)

    def split_data(self, test_size=0.2, random_state=42):

        x_train, x_test, y_train, y_test = train_test_split(
            self.img_list, self.class_list,
            test_size=test_size, random_state=random_state,
            shuffle=True, stratify=self.class_list
        )

        return x_train, x_test, y_train, y_test

    def train_model(self, x_train, y_train):

        param_grid = [
            {
                'C': [0.01, 0.05, 0.1, 1, 10, 100],
                'gamma': [0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]
            }
        ]

        classifier = svm.SVC()
        grid = GridSearchCV(classifier, param_grid)
        grid.fit(x_train, y_train)

        print(f"En iyi parametreler: {grid.best_params_}")
        self.model = grid.best_estimator_


        self.model.fit(x_train, y_train)

    def evaluate_model(self, x_test, y_test):

        predictions = self.model.predict(x_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"Model doğruluk skoru: {accuracy}")
        return accuracy

    def save_model(self):

        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model {self.model_path} dosyasına kaydedildi.")

    def run(self):

        self.load_data()
        x_train, x_test, y_train, y_test = self.split_data()
        self.train_model(x_train, y_train)
        self.evaluate_model(x_test, y_test)
        self.save_model()



