from turtle import color, home
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
from PIL import Image
import tkinter
from tkinter import filedialog
import kivy
from threading import Thread
from multiprocessing import Process

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
from utils import prediction

import os
UPDATING_FREQ = 0.001

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #decommentare per escludere la GPU
print(device_lib.list_local_devices())


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose = None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args)#, **self._kwrags)

    # def is_alive(self) -> bool:
    #     return super().is_alive()
    def join(self, *args):
        Thread.join(self, *args)
        return self._return

class DrawScreen(BoxLayout):
    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self.x0 = touch.x
            self.y0 = touch.y
            self.check = True
        else:
            self.check = False
        return super(DrawScreen, self).on_touch_down(touch)
    
    def on_touch_up(self, touch):
        self.check = False
        return super().on_touch_up(touch)

class Home (BoxLayout):
    
    Config.set('graphics', 'width', '1200')
    Config.set('graphics', 'height', '800')
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

    def __init__(self, **kwargs):
        self.loading_bar = 0
        self.thread_pred_start = 0
        super().__init__(**kwargs)

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

        if self.thread_pred_start == 1:
            # if self.thread_pred.run() == None:
            if self.loading_bar < 7:
                self.msg.text = self.msg.text + '.'
                self.loading_bar = self.loading_bar + 1
            else:
                self.msg.text = self.msg_loading
                self.loading_bar = 0

            if self.thread_pred.is_alive() == False:
                print('in')
                self.plot_mri, self.image, self.predictions, self.areas = self.thread_pred.join()
                
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

                self.thread_pred_start = 0


    def SetupVideo(self, *args):
        self.mri.canvas.after.clear()
        self.mri.canvas.before.clear()
        self.pred.disabled = self.toggle.disabled = self.slider.disabled = self.mri.disabled = self.model.disabled = self.graph.disabled = self.play.disabled = True
        self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'
        self.sbj.press_v = True
        
    def Prediction(self, *args):

        self.slider.value = 0.0
        self.msg.text = 'I am segmenting ' + self.sbj.text + ' video'
        self.msg_loading = self.msg.text
        self.model.click = False
        self.mri.canvas.after.clear()
        self.mri.canvas.before.clear()
        self.slider.disabled = self.toggle.disabled = self.play.disabled = self.mri.disabled = self.graph.disabled = True
        self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'


        images_to_keep = 5  #all is 354
        self.slider.max = images_to_keep-1
        self.slider.min = 0

        self.thread_pred = ThreadWithReturnValue(target=prediction, args=(self.fpath, images_to_keep, model_path, self.pred_model), kwargs={})
        self.thread_pred.daemon = True
        self.thread_pred.start()
        self.thread_pred_start = 1

        # self.plot_mri, self.image, self.predictions, self.areas = prediction(fpath = self.fpath, 
        #                                                                      images_to_keep = images_to_keep, 
        #                                                                      model_path = model_path, 
        #                                                                      model_name = self.pred_model)

        # self.tot_frames = self.areas[:,0].shape[0] #extract frames from areas
        # self.play.frames = self.tot_frames-1

        
        # print(self.predictions.shape)
        # print(self.areas.shape)

        # self.slider.disabled = False
        # self.toggle.disabled = False
        # self.play.disabled = False
        # self.mri.disabled = False
        # self.graph.disabled = False

        # self.image_set = []
        # self.distances = []
        
        # self.initial_dim = 0
        # self.check_distances = False
        # self.graph_plot = [None]

        # self.counter = 0
        # needed_freq = 1/25 #frames
        # self.max_count = (needed_freq/UPDATING_FREQ)
        # texture = Texture.create(size=(256, 256), colorfmt='rgb')
        # texture.blit_buffer(self.plot_mri[0].flatten(), colorfmt='rgb', bufferfmt='ubyte')
        # with self.mri.canvas.before:
        #     self.mri.canvas.clear()
        #     Color(1, 1, 1, 1)  
        #     self.image_set.append(Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,))

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
            self.graph.y_ticks_major = float(ass_ymax)
        if  self.areas[:frame,0].shape != 0:
            self.graph.xmax = float(self.areas[:frame,0].shape[0])
            self.graph.x_ticks_major = float(self.areas[:frame,0].shape[0])
        else:
            self.graph.xmax = 1.0

        self.graph.border_color = (0.3,0.3,0.3,1)


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
