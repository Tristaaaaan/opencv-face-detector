from kivy.uix.screenmanager import Screen
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.utils import platform

import numpy as np
import cv2

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


class FirstWindow(Screen):

    Builder.load_file('libs/uix/kv/firstwindow.kv')

    # Trained Model
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # Resolution
    resolution = (640, 480)
    face_resolution = (128, 96)
    ratio = resolution[0] / face_resolution[0]
    counter = 0

    def on_tex(self, *l):
        if self._camera._buffer is None:
            return None

        super(FirstWindow, self).on_tex(*l)
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

        self.update_face_status(face_detected)

        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    # Detection
    def face_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(
            gray, (self.face_resolution[1], self.face_resolution[0]))
        faces = self.face_cascade.detectMultiScale(resized, 1.3, 2)
        if len(faces) != 0:
            face = faces[np.argmax(faces[:, 3])]
            x, y, w, h = face
            cv2.rectangle(frame, (int(x * self.ratio), int(y * self.ratio)),
                          (int((x + w) * self.ratio), int((y + h) * self.ratio)), (0, 255, 0), 2)

        return len(faces) != 0

    def update_face_status(self, detected):
        if detected:
            self.ids.category.text = "Face detected"
        else:
            self.ids.category.text = "No face detected"
