from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.lang import Builder
from jnius import autoclass
import numpy as np
import cv2
from android.permissions import request_permissions, Permission
from kivy.uix.label import Label

# Define the required permissions
permissions = [
    Permission.CAMERA,
    Permission.WRITE_EXTERNAL_STORAGE,
    Permission.READ_EXTERNAL_STORAGE
]


# Camera
CameraInfo = autoclass('android.hardware.Camera$CameraInfo')
CAMERA_INDEX = {'front': CameraInfo.CAMERA_FACING_FRONT, 'back': CameraInfo.CAMERA_FACING_BACK}
index = CAMERA_INDEX['front']


class AndroidCamera(Camera):
    Builder.load_file("myapplayout.kv")
    
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

        super(AndroidCamera, self).on_tex(*l)
        self.texture = Texture.create(size=np.flip(self.resolution), colorfmt='rgb')
        frame = self.frame_from_buf()
        self.frame_to_screen(frame)

    def frame_from_buf(self):
        w, h = self.resolution
        frame = np.frombuffer(self._camera._buffer.tostring(), 'uint8').reshape((h + h // 2, w))
        frame_bgr = cv2.cvtColor(frame, 93)
        if self.index:
            return np.flip(np.rot90(frame_bgr, 1), 1)
        else:
            return np.rot90(frame_bgr, 3)

    def frame_to_screen(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame_rgb, str(self.counter), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.counter += 1
        
        # Transfer to Screen
        face_detected = self.face_det(frame_rgb)
        
        self.parent.update_face_status(face_detected)
        
        flipped = np.flip(frame_rgb, 0)
        buf = flipped.tobytes()
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    # Detection
    def face_det(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.face_resolution[1], self.face_resolution[0]))
        faces = self.face_cascade.detectMultiScale(resized, 1.3, 2)
        if len(faces) != 0:
            face = faces[np.argmax(faces[:, 3])]
            x, y, w, h = face
            cv2.rectangle(frame, (int(x * self.ratio), int(y * self.ratio)), (int((x + w) * self.ratio), int((y + h) * self.ratio)), (0, 255, 0), 2)
            
        return len(faces) != 0




class MyLayout(BoxLayout):
    def __init__(self, **kwargs):
        super(MyLayout, self).__init__(**kwargs)
        self.face_label = Label(text="No face detected", size_hint=(None, None), pos=(10, 10), halign='center', valign='middle')

        # Add the label to the layout
        self.add_widget(self.face_label)

    def update_face_status(self, detected):
        if detected:
            self.face_label.text = "Face detected"
        else:
            self.face_label.text = "No face detected"

class MyApp(App):
    def build(self):
        # Request the required permissions when the app starts
        request_permissions(permissions)
        return MyLayout()

if __name__ == '__main__':
    MyApp().run()

