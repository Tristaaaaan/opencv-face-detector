from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.graphics.texture import Texture

from kivymd.app import MDApp
import cv2
import numpy as np
from kivy.uix.camera import Camera
from kivy.utils import platform

if platform == 'android':
    from jnius import autoclass
    from android.permissions import request_permissions, Permission

    # Define the required permissions
    request_permissions([
        Permission.CAMERA,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_EXTERNAL_STORAGE
    ])

    # Camera
    CameraInfo = autoclass('android.hardware.Camera$CameraInfo')
    CAMERA_INDEX = {'front': CameraInfo.CAMERA_FACING_FRONT,
                    'back': CameraInfo.CAMERA_FACING_BACK}
    index = CAMERA_INDEX['front']


class AndroidCamera(Camera):
    net = cv2.dnn.readNet('best.onnx')
    # Define colors for each class
    class_colors = [(0, 255, 0)]
    classes = ["coffee berry"]

    # Resolution
    resolution = (640, 480)
    face_resolution = (128, 96)
    ratio = resolution[0] / face_resolution[0]
    counter = 0

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        super(AndroidCamera, self).on_tex(*l)
        self.texture = Texture.create(
            size=np.flip(self.resolution), colorfmt='rgb')
        frame = self.frame_from_buf()
        self.frame_to_screen(frame)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(),
                              'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        if self.index:
            return np.flip(np.rot90(frame_bgr, 1), 1)
        else:
            return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame_rgb, str(self.counter), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.counter += 1

        # Transfer to Screen
        face_detected = self.face_det(frame_rgb)

        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    # Detection

    def face_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (self.face_resolution[1], self.face_resolution[0]))

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1/255, size=resized, mean=[0, 0, 0], swapRB=True, crop=False)

        self.net.setInput(blob)
        detections = self.net.forward()[0]

        classes_ids = [0]
        confidences = []
        boxes = []
        rows = detections.shape[0]

        img_width, img_height = frame.shape[1], frame.shape[0]
        x_scale = img_width / 640
        y_scale = img_height / 640

        for i in range(rows):
            row = detections[i]
            confidence = row[4]
            if confidence > 0.5:
                classes_score = row[5:]
                ind = np.argmax(classes_score)
                if classes_score[ind] > 0.5:
                    classes_ids.append(ind)
                    confidences.append(confidence)
                    cx, cy, w, h = row[:4]
                    x1 = int((cx - w / 2) * x_scale)
                    y1 = int((cy - h / 2) * y_scale)
                    width = int(w * x_scale)
                    height = int(h * y_scale)
                    box = np.array([x1, y1, width, height])
                    boxes.append(box)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)

        for i in indices:
            x1, y1, w, h = boxes[i]
            label = self.classes[classes_ids[i]]
            conf = confidences[i]
            # Added a space before {:.2f}
            text = label + " {:.2f}".format(conf)
            # Get the color for the class
            class_color = self.class_colors[classes_ids[i]]

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), class_color, 2)
            cv2.putText(frame, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)


class FirstWindow(Screen):
    Builder.load_file('firstwindow.kv')


class WindowManager(ScreenManager):
    pass


class rawApp(MDApp):
    def build(self):
        return WindowManager()


if __name__ == '__main__':
    rawApp().run()
