import tensorflow as tf
import numpy as np
import os
import time
from utils import video_load
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib 
from tensorflow.python.framework.ops import disable_eager_execution
from openvino.inference_engine import IECore, Blob, TensorDesc


os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #decommentare per escludere la GPU
print(device_lib.list_local_devices())
#disable_eager_execution()

IMAGES = r'C:\Users\matte\Desktop\VTSTool_DIR\Video\S_00001\MICROSCOPIC\00001_MICROSCOPIC_01.avi'
#MODEL = r'C:\Users\matte\Desktop\VTSTool_DIR\Models\IMUNetAtt_DiceCEFocalTopK.h5' #.h5
MODEL = r'C:\Users\matte\Desktop\VTSTool_DIR\Models\IMUNetAtt_DiceCEFocalTopK' #.pb
### to produce openvino optimization write on dos:
###   mo --saved_model_dir C:\Users\matte\Desktop\VTSTool_DIR\Models\QTNet_DiceCEFocal --output_dir C:\Users\matte\Desktop\VTSTool_DIR\Models\ --input_shape [1,256,256,1] --data_type FP32 --layout NHWC --log_level DEBUG

plot_mri, image = video_load(IMAGES,5)
print(image.numpy().shape)
image = tf.transpose(image, [0,3,1,2])


input_shape_str = str([1,image.shape[0], image.shape[1], image.shape[2]]).replace(' ','')
output_dir = r'C:\Users\matte\Desktop\VTSTool_DIR\Models'

# cmd = 'python "C:\Users\matte\Desktop\VTS_Venv\Lib\site-packages\openvino\tools\mo\mo_onnx.py" \
#       --saved_model_dir '+MODEL+' --output_dir '+output_dir+' --input_shape '+input_shape_str+' --data_type FP32 --log_level DEBUG'
# # import os
# os.system(cmd)


#model = tf.keras.models.load_model(MODEL, compile = False)

plugin_dir = None
model_xml = r'C:\Users\matte\Desktop\VTSTool_DIR\Models\IMUNetAtt_DiceCEFocalTopK\saved_model.xml'
model_bin = r'C:\Users\matte\Desktop\VTSTool_DIR\Models\IMUNetAtt_DiceCEFocalTopK\saved_model.bin'
pred = np.zeros([image.shape[0], 7, image.shape[2], image.shape[3]])

# plugin = IEPlugin("CPU", plugin_dirs=plugin_dir)
ie_core_handler = IECore()
# versions = ie.get_versions("CPU")
# Read IR
net = ie_core_handler.read_network(model=model_xml, weights=model_bin)
exec_net = ie_core_handler.load_network(net, device_name='CPU', num_requests=1)
#inference_request = exec_net.requests[0]
input_blob = next(iter(net.input_info.keys()))
out_blob = next(iter(net.outputs))
net.batch_size = len([0])
net.input_info[input_blob].input_data.shape
for i in range(image.shape[0]):
    inp = np.expand_dims(image[i], 0)
    res = exec_net.infer(inputs={input_blob: inp})
    pred[i,:,:,:] = res[out_blob]
    print(i)
plt.imshow(pred[0,0,:,:])
plt.show()

print(pred.shape)



##H5
# time_before=time.time()
# print(tf. executing_eagerly())
# predictions = model.predict(image)
# print(tf. executing_eagerly())
# time_after = time.time()
# tot_time = time_after - time_before
# print('tot_time:', tot_time)