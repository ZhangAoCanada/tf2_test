import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras as k
import numpy as np
import os

class TestModel(tf.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.initializer = tf.initializers.GlorotNormal()
        self.filters = []
      
    def conv3D(self, x, filter_shape, output_ch, strides, \
                padding="SAME"):
        input_ch = x.shape[-1]
        self.filters.append(self.getFilter(filter_shape, input_ch, \
                            output_ch))
        output = tf.nn.convolution(x, self.filters[-1], strides = strides,\
                                    padding = padding)
        return output

    def getFilter(self, filter_shape=[3, 3], input_ch=1, output_ch=16):
        filter_full_shape = filter_shape + [input_ch, output_ch] 
        return tf.Variable(self.initializer(filter_full_shape))
    
    def __call__(self, x):
        output = self.conv3D(x, filter_shape = [3,3,3], output_ch = 32, \
                                strides = [1,1,1], padding = "SAME")
        output = self.conv3D(output, filter_shape = [3,3,3], output_ch = 64, \
                                strides = [1,1,1], padding = "SAME")
        return output
    
    def output(self, x):
        output = self.__call__(x)
        return output

def readData(data_dir, frame_id):
    file_name = os.path.join(data_dir, "%.6d.npy"%(frame_id))
    data = np.load(file_name)
    data_real = np.expand_dims(data.real.astype(np.float32), -1)
    data_imag = np.expand_dims(data.imag.astype(np.float32), -1)
    data = np.concatenate([data_real, data_imag], -1)
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    data = data[tf.newaxis, ...]
    return data

def main():
    time_stamp = "2020-06-01-14-44-13"
    data_dir = "/DATA/%s/ral_outputs_%s/RAD_numpy"%(time_stamp, time_stamp)
    
    data_sample = readData(data_dir, 1)
    print("===== data shape is =====")
    print(data_sample.shape)

    m = TestModel()
    # features = m(data_sample)
    features = m.output(data_sample)
    print("----- output shape is -----")
    print(features.shape)
    print("=-=-= weights shapes are =-=-=")
    for i in m.trainable_variables:
        print(i.shape)

if __name__ == "__main__":
    main() 
