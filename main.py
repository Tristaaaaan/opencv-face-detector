from kivy.uix.screenmanager import ScreenManager
from libs.uix.baseclass import firstwindow

from kivymd.app import MDApp


class WindowManager(ScreenManager):
    pass


class rawApp(MDApp):

    def build(self):

        return WindowManager()


if __name__ == '__main__':
    rawApp().run()
