import cv2 as cv
import random
import os


class ParkingSpaceSelector:
    def __init__(self, video_path, mask_path, not_empty_path, empty_path):
        self.video_path = video_path
        self.mask_path = mask_path
        self.not_empty_path = not_empty_path
        self.empty_path = empty_path

        self.not_empty_selected_box = []
        self.empty_selected_box = []
        self.saved_box = []

        self.cap = cv.VideoCapture(self.video_path)
        self.mask = cv.imread(self.mask_path)
        self.gray_mask = cv.cvtColor(self.mask, cv.COLOR_BGR2GRAY)

        self.retval, self.labels, self.stats, self.centroids = cv.connectedComponentsWithStats(
            self.gray_mask, connectivity=8, ltype=cv.CV_32S
        )

    def save_image(self, frame_crop, path):

        if not os.path.exists(path):
            os.makedirs(path)
        try:
            file_path = os.path.join(path, f"{random.randint(0, 100000000000000)}.jpg")
            cv.imwrite(file_path, frame_crop)
        except Exception as e:
            print(f"Error saving image: {e}")

    def click_event(self, event, x, y, flags, param):

        if event == cv.EVENT_LBUTTONDOWN:
            for id, (x1, y1, w, h, area) in enumerate(self.stats):
                if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                    self.not_empty_selected_box.append(id)
                    self.saved_box.append(id)

        if event == cv.EVENT_RBUTTONDOWN:
            for id, (x1, y1, w, h, area) in enumerate(self.stats):
                if x1 <= x <= x1 + w and y1 <= y <= y1 + h:
                    self.empty_selected_box.append(id)
                    self.saved_box.append(id)

    def process_frame(self, frame):

        if self.empty_selected_box:
            for i in self.empty_selected_box:
                if i != 0:
                    x, y, w, h, area = self.stats[i]
                    frame_crop = frame[y:y + h, x:x + w]
                    frame_crop = cv.resize(frame_crop, (15, 15), interpolation=cv.INTER_AREA)
                    self.save_image(frame_crop, self.empty_path)

        if self.not_empty_selected_box:
            for i in self.not_empty_selected_box:
                if i != 0:
                    x, y, w, h, area = self.stats[i]
                    frame_crop = frame[y:y + h, x:x + w]
                    frame_crop = cv.resize(frame_crop, (15, 15), interpolation=cv.INTER_AREA)
                    self.save_image(frame_crop, self.not_empty_path)

        if self.saved_box:
            for i in self.saved_box:
                if i != 0:
                    x, y, w, h, area = self.stats[i]
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        self.not_empty_selected_box.clear()
        self.empty_selected_box.clear()

    def run(self):

        cv.namedWindow('Video', cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('Video', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        cv.setMouseCallback('Video', self.click_event)

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print(f"{int(len(self.saved_box) / 2)} adet resim kaydedildi.")
                break

            self.process_frame(frame)

            cv.imshow("Video", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                print(f"{int(len(self.saved_box) / 2)} adet resim kaydedildi.")
                break

        self.cap.release()
        cv.destroyAllWindows()
