from turtle import color, home
from PIL import Image
import kivy
import cv2
import numpy as np
kivy.require("2.0.0")

from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.label import Label
from kivy.graphics import Color
from kivy.uix.button import Button
from kivy.graphics import Rectangle
from kivy.properties import ObjectProperty
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot

import os

# cap = cv2.VideoCapture(r'C:\Users\matte\Videos\GOPR1095 slow.avi', ) # video salvato in cap
# frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #larghezza immagine in pixel
# frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #altezza immagine in pixel
# print(frameCount)
# print(frameWidth)


class ColLabel(Label):
    pass

class Home (BoxLayout):
    
    sbj=ObjectProperty()
    sbjtask=ObjectProperty()
    pred=ObjectProperty()
    msg=ObjectProperty()
    mri=ObjectProperty()

    global video_path
    global dir_path
    dir_path = r"C:\Users\matte\Desktop\VTSTool_DIR"
    video_path = os.path.join(dir_path, 'Video')
    
    def update(self, *args):
        sbj = os.listdir(video_path)
        self.sbj.values = sbj

        if self.sbj.press == True:
            self.msg.text = 'Select one subject'
            self.sbj.text = 'Subject'
            self.sbj.click = False
            self.sbjtask.text = "Available Task"
            self.sbjtask.click = False
            self.pred.disabled = self.sbjtask.disabled =True
            self.sbj.press = False

        if self.sbjtask.press == True:
            self.msg.text = 'Select one task for ' + self.sbj.text
            self.sbjtask.text = "Available Task"
            self.sbjtask.click = False
            self.pred.disabled = True
            self.sbjtask.press = False
            
        if self.sbj.click == True:
            self.taskdir = os.path.join(video_path, self.sbj.text)
            st = os.listdir(self.taskdir)
            self.sbjtask.disabled = False
            self.sbjtask.values = st
            self.msg.text = 'Select one task for ' + self.sbj.text
        
        if self.sbjtask.click == True:
            self.pred.disabled = False
            self.msg.text = 'Click on Predict button to start vocal tract segmentation'
            fnames = os.listdir(os.path.join(self.taskdir, self.sbjtask.text ))
            self.fname = ''
            for file in fnames:
                if file.endswith(".avi"):
                    self.fname = file
            
            self.fdir = os.path.join(self.taskdir, self.sbjtask.text)

    def Prediction(self, *args):
        self.msg.text = 'I am loading ' + self.sbj.text + ' video'
        self.sbjtask.click = False
        self.sbj.click = False
        name = 's_' + os.path.splitext(self.fname)[0] #salva in name solo il nome del video (senza l'estensione .avi)
        dataset_dir = os.path.join(dir_path, 'Dataset')
        cap = cv2.VideoCapture(os.path.join(self.fdir,self.fname)) # video salvato in cap
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        check = True
        i = 0
        images_to_keep = 354  #all is 354
        step = int((frameCount-1)/images_to_keep)
        step_i = step
        while check:
            check, arr = cap.read()
            if check and i != 0 and step == 1: #the first image is white
                frames.append(arr)
            elif check and i !=0 and i == step:
                frames.append(arr)
                step = step + step_i
            i = i+1
        self.frames = np.rot90(np.asarray(frames), 2)

        
        
        
        texture = Texture.create(size=(256, 256), colorfmt='rgb')
        texture.blit_buffer(self.frames[0].flatten(), colorfmt='rgb', bufferfmt='ubyte')
        with self.mri.canvas:
            self.mri.canvas.clear()
            Color(1., 1, 1, 1)
            Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,)



    
class VTS_ToolApp(App):
    def build(self):
        home = Home()
        Clock.schedule_interval(home.update, 0.1)
        return home

if __name__ == '__main__':
    VTS_ToolApp().run()
