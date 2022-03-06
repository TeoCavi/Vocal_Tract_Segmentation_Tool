#%%
from turtle import color, home
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
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
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics import Rectangle
from kivy.properties import ObjectProperty
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.uix.slider import Slider
from kivy.clock import Clock
from kivy_garden.graph import Graph, LinePlot
from kivy.config import Config


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #decommentare per escludere la GPU
print(device_lib.list_local_devices())
# cap = cv2.VideoCapture(r'C:\Users\matte\Videos\GOPR1095 slow.avi', ) # video salvato in cap
# frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) #larghezza immagine in pixel
# frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #altezza immagine in pixel
# print(frameCount)
# print(frameWidth)


class ColLabel(Label):
    pass

class Home (BoxLayout):
    
    Config.set('graphics', 'width', '1400')
    Config.set('graphics', 'height', '1000')
    Config.write()
    sbj=ObjectProperty()
    sbjtask=ObjectProperty()
    pred=ObjectProperty()
    msg=ObjectProperty()
    mri=ObjectProperty()

    global video_path
    global dir_path
    global model_path
    dir_path = r"C:\Users\matte\Desktop\VTSTool_DIR"
    video_path = os.path.join(dir_path, 'Video')
    model_path = os.path.join(dir_path, 'Models')
    
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
        self.slider.value = 0.0
        self.msg.text = 'I am loading ' + self.sbj.text + ' video...'
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
        self.slider.max = images_to_keep-1
        self.slider.min = 0
        step = int((frameCount-1)/images_to_keep)
        step_i = step
        self.msg.text = 'I am extracting frames...'
        while check:
            check, arr = cap.read()
            if check and i != 0 and step == 1: #the first image is white
                frames.append(arr)
            elif check and i !=0 and i == step:
                frames.append(arr)
                step = step + step_i
            i = i+1
        self.plot_mri = np.rot90(np.asarray(frames), 2)
        self.image = np.asarray(frames)

        rgb_weights = [0.2989, 0.5870, 0.1140]
        self.image = np.dot(self.image[...,:3], rgb_weights)

        lb = np.amin(self.image)
        ub = 130.89391984
        self.image = np.where(self.image < lb, lb, self.image)
        self.image = np.where(self.image > ub, ub, self.image)
        self.image = self.image - lb
        self.image /= (ub - lb)
        self.image = np.expand_dims(self.image, 3)

        model_name = 'IMUNetAtt_DiceCEFocalTopK_20220122-093857.h5'
        model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile = False)

        self.predictions = model.predict(self.image)
        print(self.predictions.shape)

        self.predictions = Home().prediction_recomposition(self.predictions, rgba = [1, 1, 1, 1], out_classes = True)

        self.slider.disabled = False
        self.toggle.disabled = False

        texture = Texture.create(size=(256, 256), colorfmt='rgb')
        texture.blit_buffer(self.plot_mri[0].flatten(), colorfmt='rgb', bufferfmt='ubyte')
        with self.mri.canvas:
            self.mri.canvas.clear()
            Color(1, 1, 1, 1)  
            Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,)

    def Plotter(self, *args):

        frame = int(self.slider.value)

        texture = Texture.create(size=(256, 256), colorfmt='rgb')
        texture.blit_buffer(self.plot_mri[frame].flatten(), colorfmt='rgb', bufferfmt='ubyte')
        textureBK = Texture.create(size=(256, 256), colorfmt='rgba')
        textureBK.blit_buffer(self.predictions[frame,:,:,0,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')
        textureUL = Texture.create(size=(256, 256), colorfmt='rgba')
        textureUL.blit_buffer(self.predictions[frame,:,:,1,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')
        textureHP = Texture.create(size=(256, 256), colorfmt='rgba')
        textureHP.blit_buffer(self.predictions[frame,:,:,2,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')
        textureSP = Texture.create(size=(256, 256), colorfmt='rgba')
        textureSP.blit_buffer(self.predictions[frame,:,:,3,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')
        textureTO = Texture.create(size=(256, 256), colorfmt='rgba')
        textureTO.blit_buffer(self.predictions[frame,:,:,4,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')
        textureLL = Texture.create(size=(256, 256), colorfmt='rgba')
        textureLL.blit_buffer(self.predictions[frame,:,:,5,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')
        textureHE = Texture.create(size=(256, 256), colorfmt='rgba')
        textureHE.blit_buffer(self.predictions[frame,:,:,6,:].flatten(), colorfmt='rgba', bufferfmt='ubyte')

        with self.mri.canvas:
            self.mri.canvas.clear()
            Color(1, 1, 1, 1)  
            Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,)
            if self.bk.state == 'down' : 
                Color(0.5, 1, 1, 0.3)
                Rectangle(texture=textureBK, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            if self.ul.state == 'down': 
                Color(1, 0.5, 1, 0.3)
                Rectangle(texture=textureUL, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            if self.hp.state == 'down': 
                Color(1, 1, 0.5, 0.3)
                Rectangle(texture=textureHP, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            if self.sp.state == 'down': 
                Color(0.5, 1, 0.5, 0.3)
                Rectangle(texture=textureSP, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            if self.to.state == 'down': 
                Color(0.5, 0.5, 1, 0.3)
                Rectangle(texture=textureTO, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            if self.ll.state == 'down': 
                Color(1, 0.5, 0.5, 0.3)
                Rectangle(texture=textureLL, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            if self.he.state == 'down': 
                Color(0.7, 1, 0.7, 0.3)
                Rectangle(texture=textureHE, pos=self.mri.pos, size=self.mri.size,)
            else:
                Color(1,1,1,0)
            
                    


  
    def prediction_recomposition(self, prediction, rgba, out_classes):

        if len(prediction.shape) == 4 and out_classes == False:
            voxel = np.zeros([prediction.shape[0],prediction.shape[1],prediction.shape[2]])

            voxel = tf.math.argmax(prediction, axis = 3)
            voxel = tf.where(tf.equal(voxel, 0), tf.ones_like(voxel)*7, voxel)
            voxel = tf.expand_dims(voxel, 3)

        elif len(prediction.shape) == 3 and out_classes == False:
            voxel = np.zeros([prediction.shape[0],prediction.shape[1]])

            voxel = tf.math.argmax(prediction, axis = 2)
            voxel = tf.where(tf.equal(voxel, 0), tf.ones_like(voxel)*7, voxel)
        
        elif len(prediction.shape) == 4 and out_classes == True:
            print('ok')
            voxel_p = np.zeros([prediction.shape[0],prediction.shape[1],prediction.shape[2], prediction.shape[3]])
            voxel = np.zeros([prediction.shape[0],prediction.shape[1],prediction.shape[2], prediction.shape[3]])
            voxel_rgba = np.zeros([prediction.shape[0],prediction.shape[1],prediction.shape[2], prediction.shape[3], 4], np.uint8)

            voxel_p = tf.math.argmax(prediction, axis = 3)
            voxel_p = tf.where(tf.equal(voxel_p, 0), tf.ones_like(voxel_p)*7, voxel_p)
            voxel[:,:,:,0] = tf.where(tf.equal(voxel_p, 7), tf.ones_like(voxel_p)*255, voxel[:,:,:,0])
            voxel[:,:,:,1] = tf.where(tf.equal(voxel_p, 1), tf.ones_like(voxel_p)*255, voxel[:,:,:,1])
            voxel[:,:,:,2] = tf.where(tf.equal(voxel_p, 2), tf.ones_like(voxel_p)*255, voxel[:,:,:,2])
            voxel[:,:,:,3] = tf.where(tf.equal(voxel_p, 3), tf.ones_like(voxel_p)*255, voxel[:,:,:,3])
            voxel[:,:,:,4] = tf.where(tf.equal(voxel_p, 4), tf.ones_like(voxel_p)*255, voxel[:,:,:,4])
            voxel[:,:,:,5] = tf.where(tf.equal(voxel_p, 5), tf.ones_like(voxel_p)*255, voxel[:,:,:,5])
            voxel[:,:,:,6] = tf.where(tf.equal(voxel_p, 6), tf.ones_like(voxel_p)*255, voxel[:,:,:,6])


            voxel_rgba[:,:,:,0,0] = voxel[:,:,:,0]*rgba[0]
            voxel_rgba[:,:,:,0,1] = voxel[:,:,:,0]*rgba[1]
            voxel_rgba[:,:,:,0,2] = voxel[:,:,:,0]*rgba[2]
            voxel_rgba[:,:,:,0,3] = voxel[:,:,:,0]*rgba[3]
            voxel_rgba[:,:,:,1,0] = voxel[:,:,:,1]*rgba[0]
            voxel_rgba[:,:,:,1,1] = voxel[:,:,:,1]*rgba[1]
            voxel_rgba[:,:,:,1,2] = voxel[:,:,:,1]*rgba[2]
            voxel_rgba[:,:,:,1,3] = voxel[:,:,:,1]*rgba[3]
            voxel_rgba[:,:,:,2,0] = voxel[:,:,:,2]*rgba[0]
            voxel_rgba[:,:,:,2,1] = voxel[:,:,:,2]*rgba[1]
            voxel_rgba[:,:,:,2,2] = voxel[:,:,:,2]*rgba[2]
            voxel_rgba[:,:,:,2,3] = voxel[:,:,:,2]*rgba[3]
            voxel_rgba[:,:,:,3,0] = voxel[:,:,:,3]*rgba[0]
            voxel_rgba[:,:,:,3,1] = voxel[:,:,:,3]*rgba[1]
            voxel_rgba[:,:,:,3,2] = voxel[:,:,:,3]*rgba[2]
            voxel_rgba[:,:,:,3,3] = voxel[:,:,:,3]*rgba[3]            
            voxel_rgba[:,:,:,4,0] = voxel[:,:,:,4]*rgba[0]
            voxel_rgba[:,:,:,4,1] = voxel[:,:,:,4]*rgba[1]
            voxel_rgba[:,:,:,4,2] = voxel[:,:,:,4]*rgba[2]
            voxel_rgba[:,:,:,4,3] = voxel[:,:,:,4]*rgba[3]
            voxel_rgba[:,:,:,5,0] = voxel[:,:,:,5]*rgba[0]
            voxel_rgba[:,:,:,5,1] = voxel[:,:,:,5]*rgba[1]
            voxel_rgba[:,:,:,5,2] = voxel[:,:,:,5]*rgba[2]
            voxel_rgba[:,:,:,5,3] = voxel[:,:,:,5]*rgba[3]
            voxel_rgba[:,:,:,6,0] = voxel[:,:,:,6]*rgba[0]
            voxel_rgba[:,:,:,6,1] = voxel[:,:,:,6]*rgba[1]
            voxel_rgba[:,:,:,6,2] = voxel[:,:,:,6]*rgba[2]
            voxel_rgba[:,:,:,6,3] = voxel[:,:,:,6]*rgba[3]

            voxel_rgba = np.rot90(voxel_rgba, 2)

        return voxel_rgba


    
class VTS_ToolApp(App):
    def build(self):
        home = Home()
        Clock.schedule_interval(home.update, 0.1)
        return home

if __name__ == '__main__':
    VTS_ToolApp().run()
