# seleksi objek dari video
import cv2
import numpy as np

from keras.models import load_model
from keras.preprocessing.image import load_img

from keras.models import load_model
from keras.models import model_from_json


class staticROI(object):

    def __init__(self):
        self.capture = cv2.VideoCapture(2)

        # Bounding box reference points and boolean if we are extracting coordinates
        self.image_coordinates = []
        self.extract = False
        self.selected_ROI = False

        self.update()

    def update(self):
        while True:
            if self.capture.isOpened():
                # Read frame
                (self.status, self.frame) = self.capture.read()
                cv2.imshow('image', self.frame)
                key = cv2.waitKey(2)

                # Crop image
                if key == ord('c'):
                    self.clone = self.frame.copy()
                    cv2.namedWindow('image')
                    cv2.setMouseCallback('image', self.extract_coordinates)
                    while True:
                        key = cv2.waitKey(2)
                        cv2.imshow('image', self.clone)

                        # Crop and display cropped image
                        if key == ord('s'):
                            self.crop_ROI()
                            self.show_cropped_ROI()

                        # Resume video
                        if key == ord('r'):
                            break
                # Close program with keyboard 'q'
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    exit(1)
            else:
                pass

    def extract_coordinates(self, event, x, y, flags, parameters):
        # Record starting (x,y) coordinates on left mouse button click
        if event == cv2.EVENT_LBUTTONDOWN:
            self.image_coordinates = [(x, y)]
            self.extract = True

        # Record ending (x,y) coordintes on left mouse bottom release
        elif event == cv2.EVENT_LBUTTONUP:
            self.image_coordinates.append((x, y))
            self.extract = False

            self.selected_ROI = True

            # Draw rectangle around ROI
            cv2.rectangle(
                self.clone, self.image_coordinates[0], self.image_coordinates[1], (0, 255, 0), 2)

        # Clear drawing boxes on right mouse button click
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.clone = self.frame.copy()
            self.selected_ROI = False

    def crop_ROI(self):
        if self.selected_ROI:
            self.cropped_image = self.frame.copy()

            x1 = self.image_coordinates[0][0]
            y1 = self.image_coordinates[0][1]
            x2 = self.image_coordinates[1][0]
            y2 = self.image_coordinates[1][1]

            self.cropped_image = self.cropped_image[y1:y2, x1:x2]
            self.save_image = self.cropped_image.copy()

            print('Cropped image: {} {}'.format(
                self.image_coordinates[0], self.image_coordinates[1]))
        else:
            print('Select ROI to crop before cropping')

    def show_cropped_ROI(self):
        # mengambil model JST dari file yml
        yaml_file = open('model6_saved.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_json(loaded_model_yaml)

        # mengambil bobot JST
        loaded_model.load_weights("model6_saved.h5")

        # kompilasi model JST
        loaded_model.compile(
            optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # cv2.imwrite(f'C:/SEC/crop_image.jpg', self.save_image)
        # image2 = load_img(r'C:/SEC/crop_image.jpg', target_size=(224, 224))
        cv2.imwrite(f'crop/crop_image.jpg', self.save_image)
        image2 = load_img(r'crop/crop_image.jpg', target_size=(224, 224))
        img2 = np.array(image2)
        img2 = img2 / 255.0
        img2 = img2.reshape(1, 224, 224, 3)
        pred = loaded_model.predict(img2)
        label = np.argmax(pred)

        if label == 0:
            crop_img = cv2.putText(self.cropped_image, 'headphone', (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow("Prediksi : headphone", crop_img)
        elif label == 1:
            crop_img = cv2.putText(self.cropped_image, 'keyboard', (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow("Prediksi : keyboard", crop_img)
        elif label == 2:
            crop_img = cv2.putText(self.cropped_image, 'mouse', (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow("Prediksi : mouse", crop_img)
        elif label == 3:
            crop_img = cv2.putText(self.cropped_image, 'pena', (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Prediksi : pena", crop_img)
        elif label == 4:
            crop_img = cv2.putText(self.cropped_image, 'botol plastik', (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow("Prediksi : Plastik botol", crop_img)
        else:
            crop_img = cv2.putText(self.cropped_image, 'tidak dikenali', (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.imshow("Prediksi : Null", crop_img)

        # crop_img = cv2.putText(self.cropped_image, '', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA )
        # cv2.imshow(str(label), crop_img)
        #crop_img = cv2.putText(self.cropped_image, '', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA )
        #cv2.imshow(("Prediksi : " +str(pred[0][0])), crop_img)


if __name__ == '__main__':
    static_ROI = staticROI()
