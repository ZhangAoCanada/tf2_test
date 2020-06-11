import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras as k
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class TestModel(tf.Module):
    def __init__(self, input_shape, conv_channels_list, output_size):
        """
        Args:
            input_shape                 ->          [w, h, channels]
            conv_channels_list          ->          list of ints indicating 
                                                    output channels of each
                                                    convolutional layer
            outputsize                  ->          size of labels [classes, ]
        """
        super(TestModel, self).__init__()
        self.initializer = tf.initializers.GlorotNormal()
        self.input_shape = input_shape
        conv_channels_list.insert(0, input_shape[-1])
        self.conv_lists = conv_channels_list
        self.mlp_input_size = self.getMlpInputSize()
        self.output_size = output_size
        self.filters = self.getAllFilters()
        self.weights = self.getWeights(self.mlp_input_size, self.output_size)
        self.bias = self.getBias(self.output_size)

    def getMlpInputSize(self):
        mlp_w_h = self.input_shape[1]
        for i in range(len(self.conv_lists) - 1):
            mlp_w_h = mlp_w_h // 2
        return mlp_w_h * mlp_w_h * self.conv_lists[-1]
    
    def getAllFilters(self):
        self.filter_shape = (len(self.input_shape) - 1) * [3]
        self.num_conv_layers = len(self.conv_lists) - 1
        filter_list = []
        for ind in range(self.num_conv_layers):
            filter_list.append(self.getFilter(self.filter_shape, \
                                self.conv_lists[ind], self.conv_lists[ind+1]))
        return filter_list

    def getWeights(self, input_size, output_size):
        weights_shape = [input_size, output_size]
        return tf.Variable(self.initializer(weights_shape))

    def getBias(self, output_size):
        bias_shape = [output_size]
        return tf.Variable(self.initializer(bias_shape))

    def getFilter(self, filter_shape=[3, 3], input_ch=1, output_ch=16):
        filter_full_shape = filter_shape + [input_ch, output_ch] 
        return tf.Variable(self.initializer(filter_full_shape))
      
    def conv3D(self, x, filters, strides, padding="SAME"):
        output = nn.convolution(x, filters, strides = strides,\
                                    padding = padding)
        return output

    def mlp(self, x):
        output = tf.matmul(x, self.weights) + self.bias
        return output
   
    def __call__(self, x):
        output = x
        feature_maps = []
        pool_maps = []
        for conv_i in range(len(self.conv_lists)-1):
            output = self.conv3D(output, self.filters[conv_i],
                                strides = [1,1], padding = "SAME")
            feature_maps.append(tf.identity(output))
            output = nn.pool(output, [3,3], "MAX", [2,2], "SAME")
            pool_maps.append(tf.identity(output))
        output = tf.reshape(output, [output.shape[0], -1])
        output = self.mlp(output)
        return output, feature_maps, pool_maps

def getLoss(logits, labels):
    return nn.softmax_cross_entropy_with_logits(labels, logits)

def getOptimizer(lr):
    return k.optimizers.Adam(learning_rate=lr)

def getLrDecay(lr_initial, decay_steps):
    return k.optimizers.schedules.ExponentialDecay(lr_initial, decay_steps, 0.96, True)

def getMetrics():
    return k.metrics.BinaryAccuracy()
    # return k.metrics.CategoricalAccuracy()

@tf.function
def train_step(model, optimizer, data, labels):
    with tf.GradientTape() as tape:
        logits = model(data)
        loss = getLoss(logits, labels)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    display_loss = tf.math.reduce_mean(loss)
    return display_loss

@tf.function
def test_step(model, metrics, data, labels):
    logits = model(data)
    pred = nn.softmax(logits)
    _ = metrics.update_state(labels, pred) 

def trainDataGenerator(x_train, y_train, batch_size):
    total_batches = len(x_train) // batch_size
    for i in range(total_batches):
        batch_data = x_train[i*batch_size: (i+1)*batch_size]
        batch_labels = y_train[i*batch_size: (i+1)*batch_size]
        yield batch_data, batch_labels, total_batches, i

def ckFeatures(original, feature_maps, pool_maps):
    input_num = len(feature_maps[0])
    layers_num = len(feature_maps)
    feature_output = []
    pool_output = []
    for i in range(layers_num):
        features_present = []
        pool_present = []
        ori_present = []
        for j in range(int(np.sqrt(input_num))):
            feature_horizontal = []
            pool_horizontal = []
            ori_horizontal = []
            for k in range(int(np.sqrt(input_num))):
                feature_horizontal.append(feature_maps[i][j*int(np.sqrt(input_num))+k])
                pool_horizontal.append(pool_maps[i][j*int(np.sqrt(input_num))+k])
                ori_horizontal.append(original[j*int(np.sqrt(input_num))+k])
            feature_horizontal = np.concatenate(feature_horizontal, 1)
            pool_horizontal = np.concatenate(pool_horizontal, 1)
            ori_horizontal = np.concatenate(ori_horizontal, 1)
            features_present.append(feature_horizontal)
            pool_present.append(pool_horizontal)
            ori_present.append(ori_horizontal)
        features_present = np.concatenate(features_present, 0)
        pool_present = np.concatenate(pool_present, 0)
        ori_present = np.concatenate(ori_present, 0)
        feature_output.append(features_present)
        pool_output.append(pool_present)
    return ori_present, feature_output, pool_output
   
def main():
    # prepare dataset
    mnist = k.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.cast(x_train, tf.float32), tf.cast(x_test, tf.float32)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    # basic training parameters
    batch_size = 300
    epoch = 50
    lr_initial = 3e-4
    decay_steps = 1000

    # customized parameters
    input_shape = [28, 28, 1]
    conv_channels_list = [16, 32]
    output_size = y_train.shape[-1]

    # building traning testing models and etc
    m = TestModel(input_shape, conv_channels_list, output_size)
    lr = getLrDecay(lr_initial, decay_steps)
    optimizer = getOptimizer(lr)
    metrics = getMetrics()

    # tensorboard settings
    log_dir = "./log/"
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    summary_writer = tf.summary.create_file_writer(log_dir)

    # checkpoint settings
    checkpnt_dir = "./checkpoint"
    if not os.path.exists(checkpnt_dir):
        os.mkdir(checkpnt_dir)
    checkpnt_prefix = os.path.join(checkpnt_dir, "mnist_test.ckpt")
    checkpoint = tf.train.Checkpoint(model=m)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpnt_dir, max_to_keep=3)

    # restore settings
    checkpoint.restore(ckpt_manager.latest_checkpoint)
    if ckpt_manager.latest_checkpoint:
        print("Restored from {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    # # verify the restored model
    # metrics.reset_states()
    # for test_data, test_labels, _, _ in \
        # trainDataGenerator(x_test, y_test, batch_size):
        # test_step(m, metrics, x_test, y_test)
    # acc = metrics.result()
    # print("---------------------------------------------")
    # print("Accuracy of the current model:\t", acc.numpy())

    # show feature maps to see what it has learned
    x_test_extraction = x_test[:9, ...]
    logits, features, pools = m(x_test_extraction)
    pred_class = tf.math.argmax(tf.nn.softmax(logits), axis=-1)
    imgs, fmaps, pmaps = ckFeatures(x_test_extraction, features, pools)
    for i in range(len(fmaps)):
        fmaps[0] = np.squeeze(fmaps[0])
        pmaps[0] = np.squeeze(pmaps[0])
    imgs = np.squeeze(imgs)
    print("---------- class prediction ------------")
    print(pred_class)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.imshow(imgs)
    # for i in range(fmaps[0].shape[-1]):
    for j in range(fmaps[1].shape[-1]):
        ax2.clear()
        # ax2.imshow(fmaps[0][..., i])
        ax2.imshow(fmaps[1][..., j])
        fig.canvas.draw()
        plt.pause(0.5)

if __name__ == "__main__":
    main() 
