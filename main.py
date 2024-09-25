import cv2 as cv
import pickle
from Model_Train import ImageClassifier
from collect_train_data import ParkingSpaceSelector
class ParkingSpaceDetector:
    def __init__(self, video_source, mask_path, model_path):

        self.video_source = video_source
        self.mask_path = mask_path
        self.model_path = model_path
        self.model = self.load_model()
        self.mask = self.load_mask()
        self.stats = self.get_mask_stats()
        self.cap = cv.VideoCapture(self.video_source)


        cv.namedWindow('Video', cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('Video', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    def load_model(self):

        with open(self.model_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def load_mask(self):

        mask = cv.imread(self.mask_path)
        return mask

    def get_mask_stats(self):

        gray_mask = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)
        retval, labels, stats, centroids = cv.connectedComponentsWithStats(gray_mask, connectivity=8, ltype=cv.CV_32S)
        return stats

    def process_frame(self, frame):

        empty = 0
        not_empty = 0
        for i in range(1, len(self.stats)):
            x, y, w, h, area = self.stats[i]
            car_area = frame[y:y + h, x:x + w]
            car_area = cv.resize(car_area, (15, 15), cv.INTER_AREA)
            car_area = car_area.flatten()
            car_area = car_area.reshape(1, -1)
            prediction = self.model.predict(car_area)

            if prediction[0] == 1:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                not_empty += 1
            elif prediction[0] == 0:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                empty += 1

        return frame, empty, not_empty

    def display_info(self, frame, empty, not_empty):

        cv.putText(frame, f"Bos Alan: {empty}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv.putText(frame, f"Dolu Alan: {not_empty}", (10, 65), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def run(self):

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, empty, not_empty = self.process_frame(frame)
            self.display_info(frame, empty, not_empty)

            cv.imshow("Video", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv.destroyAllWindows()



if __name__ == "__main__":
    video_source = "car_park.mp4"
    mask_path = "mask.png"
    model_path = "svm_model.pkl"
    train_directory = 'train_img'
    not_empty_path = "train_img/full/"
    empty_path = "train_img/empty/"

    # Eğitim setine veri eklemek için aşşağıdaki ilk iki satırı çalıştırın.
    # parking_selector = ParkingSpaceSelector(video_source, mask_path, not_empty_path, empty_path)
    # parking_selector.run()

    # SVM modelini yeniden eğitmek için aşağıdaki satırı çalıştırın.
    # classifier = ImageClassifier(train_directory, model_path)
    # classifier.run()

    # Uygulamayı başlatmak için aşağıdaki satırı çalıştırın.
    detector = ParkingSpaceDetector(video_source, mask_path, model_path)
    detector.run()
