from turtle import color, home
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
from PIL import Image
import tkinter
from tkinter import filedialog
import kivy
import cv2
import numpy as np
# import pyautogui
kivy.require("2.0.0")

from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from kivy.uix.label import Label
from kivy.graphics import Color
from kivy.uix.button import Button
from kivy.uix.togglebutton import ToggleButton
from kivy.graphics import Rectangle, Line, Ellipse
from kivy.properties import ObjectProperty
from kivy.uix.scrollview import ScrollView
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.spinner import Spinner
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivy_garden.graph import Graph, MeshLinePlot, LinePlot
from kivy.config import Config

import os
UPDATING_FREQ = 0.001

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #decommentare per escludere la GPU
print(device_lib.list_local_devices())

class DrawScreen(BoxLayout):
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.x0 = touch.x
            self.y0 = touch.y
            self.check = True
        else:
            self.check = False

        # if touch.is_mouse_scrolling:
        #     if touch.button == 'scrolldown':
        #         if self.scale < 10:
        #             self.scale = self.scale * 1.1
        #     elif touch.button == 'scrollup':
        #         if self.scale > 1:
        #             self.scale = self.scale * 0.8
    
        return super(DrawScreen, self).on_touch_down(touch)
    
    def on_touch_up(self, touch):
        self.check = False
        return super().on_touch_up(touch)

class Home (BoxLayout):
    
    Config.set('graphics', 'width', '1200')
    Config.set('graphics', 'height', '800')
    #Config.set('graphics', 'position', 'auto')
    Config.write()
    sbj=ObjectProperty()
    sbjtask=ObjectProperty()
    pred=ObjectProperty()
    msg=ObjectProperty()
    mri=ObjectProperty()

    global dir_path
    global model_path
    dir_path = r"C:\Users\matte\Desktop\VTSTool_DIR"
    model_path = os.path.join(dir_path, 'Models')

    def update(self, *args):
        if self.sbj.press_v == True:
            self.sbj.press_v = False

            tkinter.Tk().withdraw()
            self.fpath = filedialog.askopenfilename(initialdir = dir_path,     #salva in filename il nome del video con la sua collocazione
                                            title = "Select Video",
                                            filetypes= (("avi files", "*.avi"), ("all files", "*.*"))) #find only .avi files
            Window.raise_window()

            print(self.fpath)
            if self.fpath == '':
                self.sbj.text = 'Select Video'
            else:
                name = os.path.splitext(os.path.basename(self.fpath))
                self.sbj.text = name[0]
                self.model.disabled = False
                models = os.listdir(model_path)
                self.model.values = models

        if self.model.press == True:
            self.msg.text = 'Select one Model for prediction'
            self.model.text = ""
            self.model.click = False
            self.mri.canvas.after.clear()
            self.mri.canvas.before.clear()
            self.pred.disabled = self.toggle.disabled = self.slider.disabled = self.mri.disabled = self.graph.disabled = self.play.disabled = True
            self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'
            self.model.press = False
            self.toggle.disabled
                  
        if self.model.click == True:
            self.pred.disabled = False
            self.pred_model =  self.model.text
            self.msg.text = 'Click on Predict to start Vocal Tract Segmentation'

        if self.mri.disabled == False and self.initial_dim != self.size:
            #Update is to remove blocking functions
            for el in self.image_set:
                el.pos = self.mri.pos
                el.size = self.mri.size
            

            if self.check_distances == True:

                print(self.initial_dim)
                print(self.size)
                print(self.size[0])
                print(self.initial_dim[0])
                stretch = (self.size[0]/self.initial_dim[0], self.size[1]/self.initial_dim[1])
                print(self.initial_pos)
                print(self.mri.pos)
                # delta = [self.initial_pos[0]-self.mri.pos[0], self.initial_pos[1]-self.mri.pos[1]]
                # print(delta)
                num = 0
                for ds in self.distances:
                    if (num % 2) == 0: #Ellipses
                        print(ds.pos)
                        ds.pos = (ds.pos[0]*stretch[0], ds.pos[1]*stretch[1])
                        print(ds.pos)
                    else:
                        print('before',ds.points)
                        ds.points = (ds.points[0]*stretch[0], ds.points[1]*stretch[1], ds.points[2]*stretch[0], ds.points[3]*stretch[1])
                        ds.width = ds.width*stretch[1]
                        print('after', ds.points)
                    num = num + 1
            
            self.initial_dim = self.size.copy()
        
        if self.play.state == 'down':
            self.counter = self.counter + 1
            if self.counter == self.max_count:
                if self.slider.value == self.play.frames:
                    self.slider.value = 0
                    self.play.state = 'normal'
                else:
                    self.slider.value = self.slider.value + 1
                self.counter = 0


    def SetupVideo(self, *args):
        self.mri.canvas.after.clear()
        self.mri.canvas.before.clear()
        self.pred.disabled = self.toggle.disabled = self.slider.disabled = self.mri.disabled = self.model.disabled = self.graph.disabled = self.play.disabled = True
        self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'
        self.sbj.press_v = True



        
    def Prediction(self, *args):
        self.slider.value = 0.0
        self.msg.text = 'I am loading ' + self.sbj.text + ' video...'
        self.model.click = False
        # name = 's_' + os.path.splitext(self.fname)[0] #salva in name solo il nome del video (senza l'estensione .avi)
        # dataset_dir = os.path.join(dir_path, 'Dataset')
        cap = cv2.VideoCapture(self.fpath) # video salvato in cap
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

        model_name = self.pred_model #'IMUNetAtt_DiceCEFocalTopK_20220122-093857.h5'
        model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile = False)

        print('Inizio predizioni')
        self.predictions = model.predict(self.image, batch_size = 32)
        print('fine predizioni')
        print('inizio ricomposizione')
        self.predictions, self.areas = Home().prediction_recomposition(self.predictions, rgba = [1, 1, 1, 1], out_classes = True)
        print('fine ricomposizione')

        self.tot_frames = self.areas[:,0].shape[0] #extract frames from areas
        self.play.frames = self.tot_frames-1 

        
        print(self.predictions.shape)
        print(self.areas.shape)

        self.slider.disabled = False
        self.toggle.disabled = False
        self.play.disabled = False
        self.mri.disabled = False
        self.graph.disabled = False

        self.image_set = []
        self.distances = []
        
        self.initial_dim = 0
        self.check_distances = False
        self.graph_plot = [None]

        self.counter = 0
        needed_freq = 1/25 #frames
        self.max_count = (needed_freq/UPDATING_FREQ)
        

        texture = Texture.create(size=(256, 256), colorfmt='rgb')
        texture.blit_buffer(self.plot_mri[0].flatten(), colorfmt='rgb', bufferfmt='ubyte')
        with self.mri.canvas.before:
            self.mri.canvas.clear()
            Color(1, 1, 1, 1)  
            self.image_set.append(Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,))

    def Plotter(self, clear, *args):

        print(self.play.size_hint)

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

        for g in self.graph_plot:
            self.graph.remove_plot(g)
        self.graph._clear_buffer()
        self.graph_plot = []
        ymax = []

        x = np.arange(frame+1)

        with self.mri.canvas.before:
            if clear == True:
                self.mri.canvas.after.clear()
                self.mri.canvas.before.clear()
                self.image_set = []
                self.distances = []

            Color(1, 1, 1, 1)  
            self.image_set.append(Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,))
            if self.bk.state == 'down' : 
                Color(0.5, 1, 1, 0.3)
                self.image_set.append(Rectangle(texture=textureBK, pos=self.mri.pos, size=self.mri.size,))
            if self.ul.state == 'down': 
                Color(1, 0.5, 1, 0.3)
                self.image_set.append(Rectangle(texture=textureUL, pos=self.mri.pos, size=self.mri.size,))
            if self.hp.state == 'down': 
                Color(1, 1, 0.5, 0.3)
                self.image_set.append(Rectangle(texture=textureHP, pos=self.mri.pos, size=self.mri.size,))
            if self.sp.state == 'down': 
                Color(0.5, 1, 0.5, 0.3)
                self.image_set.append(Rectangle(texture=textureSP, pos=self.mri.pos, size=self.mri.size,))
            if self.to.state == 'down': 
                Color(0.5, 0.5, 1, 0.3)
                self.image_set.append(Rectangle(texture=textureTO, pos=self.mri.pos, size=self.mri.size,))
            if self.ll.state == 'down': 
                Color(1, 0.5, 0.5, 0.3)
                self.image_set.append(Rectangle(texture=textureLL, pos=self.mri.pos, size=self.mri.size,))
            if self.he.state == 'down': 
                Color(0.7, 1, 0.7, 0.3)
                self.image_set.append(Rectangle(texture=textureHE, pos=self.mri.pos, size=self.mri.size,))

        #They must be outside of with canvas:, otherwise double line
        if self.bk.state == 'down' : 
            self.plot_bk = MeshLinePlot(color=[0.5, 1, 1, 1])
            self.plot_bk.points = [(x_0, self.areas[x_0,0].numpy()/100) for x_0 in x]
            self.graph_plot.append(self.plot_bk)
            ymax.append(int(np.max(self.areas[:,0])/100 *2))
        if self.ul.state == 'down': 
            self.plot_ul = MeshLinePlot(color=[1, 0.5, 1, 1])
            self.plot_ul.points = [(x_1, self.areas[x_1,1].numpy()/100) for x_1 in x]
            self.graph_plot.append(self.plot_ul)
            ymax.append(int(np.max(self.areas[:,1])/100 *2))
        if self.hp.state == 'down': 
            self.plot_hp = MeshLinePlot(color=[1, 1, 0.5, 1])
            self.plot_hp.points = [(x_2, self.areas[x_2,2].numpy()/100) for x_2 in x]
            self.graph_plot.append(self.plot_hp)
            ymax.append(int(np.max(self.areas[:,2])/100 *2))
        if self.sp.state == 'down': 
            Color(0.5, 1, 0.5, 0.3)
            self.plot_sp = MeshLinePlot(color=[0.5, 1, 0.5, 1])
            self.plot_sp.points = [(x_3, self.areas[x_3,3].numpy()/100) for x_3 in x]
            self.graph_plot.append(self.plot_sp)
            ymax.append(int(np.max(self.areas[:,3])/100 *2))
        if self.to.state == 'down': 
            self.plot_to = MeshLinePlot(color=[0.5, 0.5, 1, 1])
            self.plot_to.points = [(x_4, self.areas[x_4,4].numpy()/100) for x_4 in x]
            self.graph_plot.append(self.plot_to)
            ymax.append(int(np.max(self.areas[:,4])/100 *2))
        if self.ll.state == 'down': 
            self.plot_ll = MeshLinePlot(color=[1, 0.5, 0.5, 1])
            self.plot_ll.points = [(x_5, self.areas[x_5,5].numpy()/100) for x_5 in x]
            self.graph_plot.append(self.plot_ll)
            ymax.append(int(np.max(self.areas[:,5])/100 *2))
        if self.he.state == 'down': 
            self.plot_he = LinePlot(color=[0.7, 1, 0.7, 1])
            self.plot_he.points = [(x_6, self.areas[x_6,6].numpy()/100) for x_6 in x]
            self.graph_plot.append(self.plot_he)
            ymax.append(int(np.max(self.areas[:,6])/100 *2))

        if ymax:
            ass_ymax = np.max(np.asarray(ymax))
            self.graph.ymax = float(ass_ymax)
        if  self.areas[:frame,0].shape != 0:
            self.graph.xmax = float(self.areas[:frame,0].shape[0])
        else:
            self.graph.xmax = 1.0


        for g in self.graph_plot:
            self.graph.add_plot(g)

        #print(self.boxdim.size)

    def Draw(self, set = 1, *args):
        print('-------------')
        print(self.mri.x0)
        print(self.mri.y0)
        print(self.mri.check)
        print('-------------')
        with self.mri.canvas.after:
            self.d = 2
            if self.mri.state == 0:
                self.initial_dim = self.size.copy()
                self.initial_pos = self.mri.pos.copy()
                print('a', self.initial_dim)
                Color(1,1,1,1)
                self.distances.append(Ellipse(size = (self.d,self.d), pos = (self.mri.x0 , self.mri.y0 )))
                self.mri.x1 = self.mri.x0 
                self.mri.y1 = self.mri.y0 
                self.mri.state = 1
                self.check_distances = True
            elif self.mri.state == 1:
                if self.initial_dim != self.size:
                    self.mri.canvas.after.remove(self.distances[-1])
                    self.distances.pop(-1)
                Color(1,1,1,1)
                self.distances.append(Line(points = [self.mri.x1, self.mri.y1, self.mri.x0 , self.mri.y0 ], width = 1.5))
                self.mri.state = 0
                self.check_distances = True
                   
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

            areas = tf.math.count_nonzero(voxel, axis = (1,2))

        return voxel_rgba, areas
    
    def time(self, *args):
        time.sleep(1)

    
    def Play(self, *args):
        if self.slider.value+1 == self.tot_frames:
             self.slider.value = 0
        else:
            self.slider.value = self.slider.value + 1
            print(self.slider.value)
            time.sleep(1)

class VTS_ToolApp(App):
    def build(self):
        home = Home()
        Clock.schedule_interval(home.update, UPDATING_FREQ)
        return home

if __name__ == '__main__':
    VTS_ToolApp().run()
