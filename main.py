from kivy.uix.screenmanager import ScreenManager, Screen
import datetime
from kivymd.uix.pickers import MDDatePicker
from kivymd.app import MDApp
from kivymd.uix.button import MDFlatButton
from datetime import datetime
from kivy.lang import Builder
from kivy.graphics.texture import Texture
from kivy.uix.camera import Camera
from kivy.utils import platform
from os.path import dirname, join
import numpy as np
import cv2
from kivy.clock import Clock
from kivy.properties import StringProperty, ColorProperty
from kivymd.uix.card import MDCard
from kivymd.uix.label import MDLabel
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.dialog import MDDialog

if platform == 'android':
    from androidstorage4kivy import SharedStorage
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

    # Environment
    Environment = autoclass('android.os.Environment')


class AndroidCamera(Camera):

    # Trained Model
    face_cascade = cv2.CascadeClassifier('cascade.xml')

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
        faces = self.face_cascade.detectMultiScale(resized, 1.3, 2)
        if len(faces) != 0:
            face = faces[np.argmax(faces[:, 3])]
            x, y, w, h = face
            cv2.rectangle(frame, (int(x * self.ratio), int(y * self.ratio)),
                          (int((x + w) * self.ratio), int((y + h) * self.ratio)), (0, 255, 0), 2)

        return len(faces) != 0


class MainWindow(Screen):

    Builder.load_file('mainwindow.kv')

    def start_detection(self):
        self.manager.current = "first"
        self.manager.transition.direction = "up"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self.date)

    def date(self, *args):
        # Define the format you want
        date_format = "%A, %B %d, %Y"

        # Get the current date in the specified format
        current_date = datetime.now().strftime(date_format)

        self.ids.current_date.text = current_date


class FirstWindow(Screen):

    Builder.load_file('firstwindow.kv')

    def switch_to_main(self):
        self.manager.current = "main"
        self.manager.transition.direction = "right"


class HistoryWindow(Screen):

    Builder.load_file('historywindow.kv')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self.data_)

    def data_(self, *args):

        # Add more texts as needed
        generated_texts = ['Text1', 'Text2', 'Text3', 'Text3', 'Text3']

        for generated_text in generated_texts:
            self.identity = '2 Detected'
            self.background_color = (321/255, 245/255, 245/255, 1)
            self.icon = 'check'
            self.color = [0, 0, 0, 1]
            self.identity_color = [224/255, 224/255, 224/255, 1]
            self.identity_bg = [1, 1, 1, 1]
            self.id_text_color = [1, 1, 1, 1]
            self.date = 'Wednesday, November 8, 2023'

            add_bot = Chats(
                text=generated_text, identity=self.identity, background_color=self.background_color, icon=self.icon, context_color=self.color, identity_color=self.identity_color, identity_bg=self.identity_bg, id_text_color=self.id_text_color, date=self.date)

            self.ids.listexpenses.add_widget(add_bot)

    def switch_to_main(self):
        self.manager.current = "main"
        self.manager.transition.direction = "right"


class DialogContent(MDBoxLayout):

    """Customize dialog box for user to insert their expenses"""
    # Initiliaze date to the current date

    def cancel(self):
        MDApp.get_running_app().root.settings.close_dialog()

    def on_kv_post(self, base_widget):
        self.ids.seconds_bg.md_bg_color = [114/255, 89/255, 89/255, 1]
        self.ids.minutes_bg.md_bg_color = [125/255, 125/255, 125/255, 1]
        self.ids.hours_bg.md_bg_color = [125/255, 125/255, 125/255, 1]

        return super().on_kv_post(base_widget)

    def color(self, background, label):
        self.ids.seconds_bg.md_bg_color = [125/255, 125/255, 125/255, 1]
        self.ids.minutes_bg.md_bg_color = [125/255, 125/255, 125/255, 1]
        self.ids.hours_bg.md_bg_color = [125/255, 125/255, 125/255, 1]

        background.md_bg_color = [114/255, 89/255, 89/255, 1]


class SettingsWindow(Screen):

    Builder.load_file('settingswindow.kv')

    def switch_to_main(self):
        self.manager.current = "main"
        self.manager.transition.direction = "right"

    def config_length(self):
        self.task_list_dialog = MDDialog(
            title="Configure Detection Length",
            type="custom",
            size_hint_x=0.9,
            size_hint_y=None,
            content_cls=DialogContent(),
        )

        self.task_list_dialog.open()

    def close_dialog(self, *args):
        self.task_list_dialog.dismiss()


class CustomLabel(MDLabel):
    pass


class Chats(MDCard):
    text = StringProperty()
    identity = StringProperty()
    date = StringProperty()
    background_color = ColorProperty()
    icon = StringProperty()
    context_color = ColorProperty()
    identity_color = ColorProperty()
    identity_bg = ColorProperty()
    id_text_color = ColorProperty()

    def switch_screen(self):
        print('????')


class WindowManager(ScreenManager):
    pass


class rawApp(MDApp):

    def build(self):

        return WindowManager()


if __name__ == '__main__':
    rawApp().run()
