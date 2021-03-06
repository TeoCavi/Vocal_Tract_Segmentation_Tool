import tensorflow as tf
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
from openvino.inference_engine import IECore, Blob, TensorDesc


def prediction_recomposition(prediction, rgba, out_classes):

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


def video_load(fpath,images_to_keep):
    
    cap = cv2.VideoCapture(fpath) # video salvato in cap
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    check = True
    i = 0

    step = int((frameCount-1)/images_to_keep)
    step_i = step
    #self.msg.text = 'I am extracting frames...'

    #first image removal 
    while check:
        check, arr = cap.read()
        if check and i != 0 and step == 1: #the first image is white
            frames.append(arr)
        elif check and i !=0 and i == step:
            frames.append(arr)
            step = step + step_i
        i = i+1
    
    plot_mri = np.rot90(np.asarray(frames), 2) #used to build image texture for canvas
    image = np.asarray(frames) #used for prediction

    #from RGB to Luminance
    rgb_weights = [0.2989, 0.5870, 0.1140]
    image = np.dot(image[...,:3], rgb_weights)

    #input image normalization
    lb = np.amin(image)
    ub = 130.89391984
    image = np.where(image < lb, lb, image)
    image = np.where(image > ub, ub, image)
    image = image - lb
    image /= (ub - lb)
    image = np.expand_dims(image, 3)
    image = tf.cast(image, tf.float32)

    return plot_mri, image


def prediction(fpath,images_to_keep, model_path, model_name):
    
    plot_mri, image = video_load(fpath,images_to_keep)
    _, model_ext = os.path.splitext(os.path.basename(os.path.join(model_path, model_name)))

    #     # plot_mri, image = video_load(fpath,images_to_keep)
    # _, model_ext = os.path.splitext(os.path.basename(os.path.join(model_path, model_name)))

    # cap = cv2.VideoCapture(fpath) # video salvato in cap
    # frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # frames = []
    # check = True
    # i = 0

    # step = int((frameCount-1)/images_to_keep)
    # step_i = step
    # #self.msg.text = 'I am extracting frames...'

    # #first image removal 
    # while check:
    #     check, arr = cap.read()
    #     if check and i != 0 and step == 1: #the first image is white
    #         frames.append(arr)
    #     elif check and i !=0 and i == step:
    #         frames.append(arr)
    #         step = step + step_i
    #     i = i+1
    
    # plot_mri = np.rot90(np.asarray(frames), 2) #used to build image texture for canvas
    # image = np.asarray(frames) #used for prediction

    # #from RGB to Luminance
    # rgb_weights = [0.2989, 0.5870, 0.1140]
    # image = np.dot(image[...,:3], rgb_weights)

    # #input image normalization
    # lb = np.amin(image)
    # ub = 130.89391984
    # image = np.where(image < lb, lb, image)
    # image = np.where(image > ub, ub, image)
    # image = image - lb
    # image /= (ub - lb)
    # image = np.expand_dims(image, 3)
    # image = tf.cast(image, tf.float32)
    

    # predictions = interpreter.get_tensor(output_details[0]['index'])
    # print(predictions.shape)
    predictions = np.zeros([image.shape[0], image.shape[1], image.shape[2], 7])
    time_before=time.time()

    print(model_ext)

    if model_ext == '.tflite':
        interpreter = tf.lite.Interpreter(model_path = os.path.join(model_path, model_name))
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        for i in range(image.shape[0]):
            interpreter.set_tensor(input_details[0]['index'], tf.expand_dims(image[i],0))
            interpreter.invoke()
            predictions[i,:,:,:] = interpreter.get_tensor(output_details[0]['index'])
    elif model_ext == '.h5':
        model = tf.keras.models.load_model(os.path.join(model_path, model_name), compile = False)
        predictions = model.predict(image, batch_size = 32)
    elif model_ext == '': #openvino optimization
        image = tf.transpose(image, [0,3,1,2])
        model_xml = os.path.join(model_path, model_name,'saved_model.xml')
        model_bin = os.path.join(model_path, model_name,'saved_model.bin')
        print(model_xml)
        pred = np.zeros([image.shape[0], 7, image.shape[2], image.shape[3]])

        # plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
        ie_core_handler = IECore()
        # versions = ie.get_versions("CPU")
        # Read IR
        net = ie_core_handler.read_network(model=model_xml, weights=model_bin)
        exec_net = ie_core_handler.load_network(net, device_name='CPU', num_requests=1)
        input_blob = next(iter(net.input_info.keys()))
        out_blob = next(iter(net.outputs))
        net.batch_size = len([0])
        net.input_info[input_blob].input_data.shape
        for i in range(image.shape[0]):
            inp = np.expand_dims(image[i], 0)
            res = exec_net.infer(inputs={input_blob: inp})
            pred[i,:,:,:] = res[out_blob]

        predictions = tf.transpose(pred, [0,2,3,1]) 


    time_after = time.time()
    tot_time = time_after - time_before
    print('tot_time:', tot_time)

    predictions, areas = prediction_recomposition(predictions, rgba = [1, 1, 1, 1], out_classes = True)

    # x = np.arange(image.shape[0])
    # x_0 = np.ones([7,image.shape[0]])
    # x = x[None,:]*x_0
    # # x = x.astype(int)
    # y = np.transpose(areas)
    # #areas = np.stack((x,y),axis=2)
    # z = []
    # z.append(list(map(lambda x,y: (x,y) , x[0,:], y[0,:])))
    # z.append(list(map(lambda x,y: (x,y) , x[1,:], y[1,:])))
    # z.append(list(map(lambda x,y: (x,y) , x[2,:], y[2,:])))
    # z.append(list(map(lambda x,y: (x,y) , x[3,:], y[3,:])))
    # z.append(list(map(lambda x,y: (x,y) , x[4,:], y[4,:])))
    # z.append(list(map(lambda x,y: (x,y) , x[5,:], y[5,:])))
    # areas = z

    x = np.arange(image.shape[0])
    y0 = [] 
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []

    for x_0 in x:
        y0.append((x_0, areas[x_0,0].numpy())) #/100 is necessary for graph buffer
        y1.append((x_0, areas[x_0,1].numpy()))
        y2.append((x_0, areas[x_0,2].numpy()))
        y3.append((x_0, areas[x_0,3].numpy()))
        y4.append((x_0, areas[x_0,4].numpy()))
        y5.append((x_0, areas[x_0,5].numpy()))
        y6.append((x_0, areas[x_0,6].numpy()))

    return plot_mri, image, predictions, areas, y0, y1, y2, y3, y4, y5, y6
