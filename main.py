from turtle import color, home, pos
import time
from cv2 import sqrt
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
from PIL import Image
import tkinter
from tkinter import Variable, filedialog
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #decommentare per escludere la GPU
print(device_lib.list_local_devices())

DEF_UPDATE_FREQ = 0.001
PRED_FREQ = 0.5
FRAME_TO_EXTRACT = 6 #max 354
VIDEO_FREQ = 1/25 #frames

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose = None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args)#, **self._kwrags)

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
        #init loading bar
        self.loading_bar = 0 
        self.load = 0
        #init 
        self.thread_pred_start = 0
        #init canvas mri
        self.loading_fig = []
        #init play button
        self.counter = 0
        #init graph max
        self.ymax = []
        self.graph_plot = []
        super().__init__(**kwargs)
        self.clock = Clock.schedule_interval(self.update, DEF_UPDATE_FREQ)

    def update(self, *args):
        if self.sbj.press_v == True:
            self.sbj.press_v = False
            self.model.click = False

            tkinter.Tk().withdraw()
            self.fpath = filedialog.askopenfilename(initialdir = dir_path,     #salva in filename il nome del video con la sua collocazione
                                            title = "Select Video",
                                            filetypes= (("avi files", "*.avi"), ("all files", "*.*"))) #find only .avi files
            Window.raise_window()

            if self.fpath == '':
                self.sbj.text = 'Select Video'
                self.msg.text = 'Select video and model'
            else:
                name = os.path.splitext(os.path.basename(self.fpath))
                self.sbj.text = name[0]
                self.model.disabled = False
                models = os.listdir(model_path)
                self.model.values = models
                self.msg.text = 'Select one Model for prediction'

        if self.model.press == True:
            self.counter = 0
            self.ymax = []
            self.loading_bar = 0 
            self.load = 0
            self.model.text = ""
            self.model.click = False
            self.mri.canvas.after.clear()
            self.mri.canvas.before.clear()
            for g in self.graph_plot:
                self.graph.remove_plot(g)
            self.pred.disabled = self.toggle.disabled = self.slider.disabled = self.mri.disabled = self.play.disabled = self.switch.disabled = True
            self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'
            self.bkg.state = self.ulg.state = self.hpg.state = self.spg.state = self.tog.state = self.llg.state = self.heg.state = 'normal'

            self.model.press = False
            if self.model.text == '':
                self.model.text = 'Select Model'
                self.msg.text = 'Select one Model for prediction'
                self.model.click = False
                  
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
                stretch = (self.size[0]/self.initial_dim[0], self.size[1]/self.initial_dim[1])
                ds = self.distances
                for i in range(0, len(self.distances), 3):
                    #Ellipses
                    ds[i].pos = (ds[i].pos[0]*stretch[0], ds[i].pos[1]*stretch[1])

                    #Lines
                    ds[i+1].points = (ds[i+1].points[0]*stretch[0], ds[i+1].points[1]*stretch[1], ds[i+1].points[2]*stretch[0], ds[i+1].points[3]*stretch[1])
                    ds[i+1].width = ds[i+1].width*stretch[1]

                    #Labels
                    ds[i+2].pos = (ds[i+2].pos[0]*stretch[0], ds[i+2].pos[1]*stretch[1])

            self.initial_dim = self.size.copy()
        
        if self.play.state == 'down':
            # self.counter = self.counter + 1
            # if self.counter == self.max_count:
            #     if self.slider.value == self.play.frames:
            #         self.slider.value = 0
            #         self.play.state = 'normal'
            #     else:
            #         self.slider.value = self.slider.value + 1
            #     self.counter = 0
            if self.counter == 0:
                self.slider.value = 0
                self.clock.cancel()
                self.clock = Clock.schedule_interval(self.update, VIDEO_FREQ)
                self.counter = 1
            else:
                if self.slider.value == self.play.frames:
                    #self.slider.value = 0
                    self.counter = 0
                    self.play.state = 'normal'
                    self.clock.cancel()
                    self.clock = Clock.schedule_interval(self.update, DEF_UPDATE_FREQ)
                else:
                    self.slider.value = self.slider.value + 1


        if self.thread_pred_start == 1:
            
            #loading bar update
            self.loading_bar = self.loading_bar + 1
            if self.loading_bar == 1:
                self.load = self.load+45
                with self.textbar.canvas:
                    self.textbar.canvas.clear()
                    r1 = 0.7*self.textbar.height
                    r2 = 0.9*self.textbar.height

                    Color(0.12,0.18,0.20,1)
                    if self.load <= 360:
                        Ellipse(pos = (self.textbar.pos[0]+(self.textbar.width/2)-r2/2, 
                                        self.textbar.pos[1]+(self.textbar.height/2)-r2/2), 
                                        size= (r2,r2), angle_start = 0, 
                                        angle_end = self.load)
                    elif self.load > 360 and self.load <= 720:
                        Ellipse(pos = (self.textbar.pos[0]+(self.textbar.width/2)-r2/2, 
                                        self.textbar.pos[1]+(self.textbar.height/2)-r2/2), 
                                        size= (r2,r2), angle_start = self.load-360, 
                                        angle_end = 360)
                        if self.load == 720:
                            self.load = 0
                    Color(0.87, 0.91, 0.92, 0.7 )
                    Ellipse(pos = (self.textbar.pos[0]+(self.textbar.width/2)-r1/2, 
                            self.textbar.pos[1]+(self.textbar.height/2)-r1/2),
                            size = (r1,r1),
                            angle_start = 0,
                            angle_end = 360,)
                self.loading_bar = 0

            if self.thread_pred.is_alive() == False:
                with self.textbar.canvas:
                    self.textbar.canvas.clear()
                self.plot_mri, self.image, self.predictions, self.areas, self.y0, self.y1, self.y2, self.y3, self.y4, self.y5, self.y6 = self.thread_pred.join()
                self.clock.cancel()
                self.clock = Clock.schedule_interval(self.update, DEF_UPDATE_FREQ) 
                self.msg.text = self.sbj.text + ' Segmentation'

                #graph plot production
                self.tot_frames = self.areas[:,0].shape[0]
                self.play.frames = self.tot_frames-1
                self.plot_bk = MeshLinePlot(color=[0, 0.4, 0.2, 1])
                self.plot_bk.points = self.y0
                self.plot_ul = MeshLinePlot(color=[0, 0, 1, 1])
                self.plot_ul.points = self.y1
                self.plot_hp = MeshLinePlot(color=[1, 0, 0, 1])
                self.plot_hp.points = self.y2
                self.plot_sp = MeshLinePlot(color=[0.9, 0.7, 0, 1])
                self.plot_sp.points = self.y3
                self.plot_to = MeshLinePlot(color=[1, 0.07, 0.7, 1])
                self.plot_to.points = self.y4
                self.plot_ll = MeshLinePlot(color=[0.6, 0.17, 0.93, 1])
                self.plot_ll.points = self.y5
                self.plot_he = LinePlot(color=[0.69, 0.13, 0.13, 1])
                self.plot_he.points = self.y6

                self.slider.disabled = False
                self.toggle.disabled = False
                self.switch.disabled = False
                self.play.disabled = False
                self.mri.disabled = False
                

                self.image_set = []
                self.distances = []
                
                self.initial_dim = 0
                self.check_distances = False
                self.graph_plot = [] #####

                self.max_count = (VIDEO_FREQ/DEF_UPDATE_FREQ)
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
        self.pred.disabled = self.toggle.disabled = self.slider.disabled = self.mri.disabled = self.model.disabled = self.play.disabled = self.switch.disabled = True
        self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'
        self.bkg.state = self.ulg.state = self.hpg.state = self.spg.state = self.tog.state = self.llg.state = self.heg.state = 'normal'
        self.sbj.press_v = True
        self.counter = 0
        self.ymax = []
        self.loading_bar = 0 
        self.load = 0

        for g in self.graph_plot:
            self.graph.remove_plot(g)
        
    def Prediction(self, *args):
        self.counter = 0
        self.ymax = []
        self.loading_bar = 0 
        self.load = 0
        self.slider.value = 0.0
        self.msg.text = 'I am segmenting ' + self.sbj.text + ' video'
        self.msg_loading = self.msg.text
        self.model.click = False
        self.mri.canvas.after.clear()
        self.mri.canvas.before.clear()
        self.slider.disabled = self.toggle.disabled = self.play.disabled = self.mri.disabled = self.switch.disabled = True
        self.bk.state = self.ul.state = self.hp.state = self.sp.state = self.to.state = self.ll.state = self.he.state = 'normal'
        self.bkg.state = self.ulg.state = self.hpg.state = self.spg.state = self.tog.state = self.llg.state = self.heg.state = 'normal'


        images_to_keep = FRAME_TO_EXTRACT  #all is 354
        self.slider.max = images_to_keep-1
        self.slider.min = 0
        self.clock.cancel()
        self.clock = Clock.schedule_interval(self.update, PRED_FREQ)
        self.thread_pred = ThreadWithReturnValue(target=prediction, args=(self.fpath, images_to_keep, model_path, self.pred_model), kwargs={})
        self.thread_pred.daemon = True
        self.thread_pred.start()
        self.thread_pred_start = 1

    def Plotter(self, clear, *args):
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

        with self.mri.canvas.before:
            if clear == True:
                self.mri.canvas.after.clear()
                self.mri.canvas.before.clear()
                self.image_set = []
                self.distances = []

            Color(1, 1, 1, 1)  
            self.image_set.append(Rectangle(texture=texture, pos=self.mri.pos, size=self.mri.size,))
            if self.bk.state == 'down' : 
                Color(0, 1, 0, 0.3)
                #Color(0.5, 1, 1, 0.3)
                self.image_set.append(Rectangle(texture=textureBK, pos=self.mri.pos, size=self.mri.size,))
            if self.ul.state == 'down': 
                Color(0, 0, 1, 0.3)
                #Color(1, 0.5, 1, 0.3)
                self.image_set.append(Rectangle(texture=textureUL, pos=self.mri.pos, size=self.mri.size,))
            if self.hp.state == 'down': 
                Color(1, 0, 0, 0.3)
                #Color(1, 1, 0.5, 0.3)
                self.image_set.append(Rectangle(texture=textureHP, pos=self.mri.pos, size=self.mri.size,))
            if self.sp.state == 'down': 
                Color(1, 1, 0, 0.3)
                #Color(0.5, 1, 0.5, 0.3)
                self.image_set.append(Rectangle(texture=textureSP, pos=self.mri.pos, size=self.mri.size,))
            if self.to.state == 'down': 
                Color(1, 0.07, 0.7, 0.3)
                #Color(0.5, 0.5, 1, 0.3)
                self.image_set.append(Rectangle(texture=textureTO, pos=self.mri.pos, size=self.mri.size,))
            if self.ll.state == 'down': 
                Color(0.6, 0.17, 0.93, 0.3)
                #Color(1, 0.5, 0.5, 0.3)
                self.image_set.append(Rectangle(texture=textureLL, pos=self.mri.pos, size=self.mri.size,))
            if self.he.state == 'down': 
                Color(0.69, 0.13, 0.13, 0.3)
                #Color(0.7, 1, 0.7, 0.3)
                self.image_set.append(Rectangle(texture=textureHE, pos=self.mri.pos, size=self.mri.size,))

        # if self.graph_plot:
        #     for g in self.graph_plot:
        #         self.graph.remove_plot(g)
            
        # self.graph_plot = []

        self.graph.border_color = (0.12,0.18,0.20,1)
        self.ymax = []

        if self.slider.value == 0:
            for g in self.graph_plot:
                self.graph.remove_plot(g)
            self.graph_plot = []
            
        if self.bkg.state == 'down':
            if self.graph_plot.count(self.plot_bk) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_bk)
                self.graph_plot.append(self.plot_bk)
            self.bk_max = int(np.max(self.areas[:,0])/100 *2)
            self.ymax.append(self.bk_max)
        elif self.bkg.state == 'normal':
            if self.graph_plot.count(self.plot_bk) == 1:
                self.graph.remove_plot(self.plot_bk)
                self.graph_plot.remove(self.plot_bk)

        if self.ulg.state == 'down'  :
            if self.graph_plot.count(self.plot_ul) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_ul)
                self.graph_plot.append(self.plot_ul)
            self.ul_max = int(np.max(self.areas[:,1])/100 *2)
            self.ymax.append(self.ul_max)
        elif self.ulg.state == 'normal' :
            if self.graph_plot.count(self.plot_ul) == 1:
                self.graph.remove_plot(self.plot_ul)
                self.graph_plot.remove(self.plot_ul)

        if self.hpg.state == 'down'  :
            if self.graph_plot.count(self.plot_hp) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_hp)
                self.graph_plot.append(self.plot_hp)
            self.hp_max = int(np.max(self.areas[:,2])/100 *2)
            self.ymax.append(self.hp_max)
        elif self.hpg.state == 'normal' :
            if self.graph_plot.count(self.plot_hp) == 1:
                self.graph.remove_plot(self.plot_hp)
                self.graph_plot.remove(self.plot_hp)

        if self.spg.state == 'down':
            if self.graph_plot.count(self.plot_sp) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_sp)
                self.graph_plot.append(self.plot_sp)
            self.sp_max = int(np.max(self.areas[:,3])/100 *2)
            self.ymax.append(self.sp_max)
        elif self.spg.state == 'normal' :
            if self.graph_plot.count(self.plot_sp) == 1:
                self.graph.remove_plot(self.plot_sp)
                self.graph_plot.remove(self.plot_sp)
        
        if self.tog.state == 'down':
            if self.graph_plot.count(self.plot_to) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_to)
                self.graph_plot.append(self.plot_to)
            self.to_max = int(np.max(self.areas[:,4])/100 *2)
            self.ymax.append(self.to_max)
        elif self.tog.state == 'normal' :
            if self.graph_plot.count(self.plot_to) == 1:
                self.graph.remove_plot(self.plot_to)
                self.graph_plot.remove(self.plot_to)
        
        if self.llg.state == 'down':
            if self.graph_plot.count(self.plot_ll) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_ll)
                self.graph_plot.append(self.plot_ll)
            self.ll_max = int(np.max(self.areas[:,5])/100 *2)
            self.ymax.append(self.ll_max)
        elif self.llg.state == 'normal' :
            if self.graph_plot.count(self.plot_ll) == 1:
                self.graph.remove_plot(self.plot_ll)
                self.graph_plot.remove(self.plot_ll)

        if self.heg.state == 'down':
            if self.graph_plot.count(self.plot_he) == 0 and self.slider.value != 0:
                self.graph.add_plot(self.plot_he)
                self.graph_plot.append(self.plot_he)
            self.he_max = int(np.max(self.areas[:,6])/100 *2)
            self.ymax.append(self.he_max)
        elif self.heg.state == 'normal' :
            if self.graph_plot.count(self.plot_he) == 1:
                self.graph.remove_plot(self.plot_he)
                self.graph_plot.remove(self.plot_he)

        self.past_slider_value = self.slider.value

        if self.ymax:
            ass_ymax = np.max(np.asarray(self.ymax))
            self.graph.ymax = float(ass_ymax)
        else:
            self.graph.ymax = 10.0

        if  self.areas[:frame,0].shape != 0:
            self.graph.xmax = float(self.areas[:frame,0].shape[0])
            if frame == 0:
                self.graph.x_ticks_major = float(self.areas[:frame,0].shape[0])/1
            elif frame > 0 and frame < 20:
                self.graph.x_ticks_major = float(self.areas[:frame,0].shape[0])/frame
            elif frame >= 10:
                self.graph.x_ticks_major = 20
        else:
            self.graph.x_ticks_major = 1
            self.graph.xmax = 1.0

        # for g in self.graph_plot:
        #     self.graph.add_plot(g)

    def Draw(self, set = 1, *args):
        with self.mri.canvas.after:
            self.d = 2
            if self.mri.state == 0:
                self.initial_dim = self.size.copy()
                self.initial_pos = self.mri.pos.copy()
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

                #evaluating distance in 256x256 gridspace
                p1 = (self.mri.x1,self.mri.y1)
                p2 = (self.mri.x0,self.mri.y0)

                dx = abs(p1[0]-p2[0])
                dy = abs(p1[1]-p2[1])
                dx_256 = (256*dx)/self.mri.size[0]
                dy_256 = (256*dy)/self.mri.size[1]
                d_xy_256 = sqrt(pow(dx_256,2)+pow(dy_256,2))[0]
                # #diag_frame = sqrt(pow(self.size[0],2)+pow(self.size[1],2))
                # d_ab_norm = round(np.max(((256*d_ab)/diag_frame)[0]),1)
                d_xy_mm = round(np.max(d_xy_256*(1.6)),1)
                self.distances.append(Label(text = '{}mm'.format(d_xy_mm), size = (16,24), pos = (self.mri.x0,self.mri.y0), valign = 'bottom', halign = 'left'))

                self.mri.state = 0
                self.check_distances = True
    
    def time(self, *args):
        time.sleep(1)

    
    def Play(self, *args):
        if self.slider.value+1 == self.tot_frames:
             self.slider.value = 0
        else:
            self.slider.value = self.slider.value + 1

class VTS_ToolApp(App):
    def build(self):
        home = Home()
        #TimeSchedule().start(DEF_UPDATE_FREQ)
        return home

if __name__ == '__main__':
    VTS_ToolApp().run()
