import os
##### set specific gpu #####
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras as k
import numpy as np
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
        final_feature_size = []
        for j in range(len(self.input_shape) - 1):
            current_shape = self.input_shape[j]
            for i in range(len(self.conv_lists) - 1):
                current_shape = current_shape // 2
                if current_shape == 0:
                    current_shape += 1
            final_feature_size.append(current_shape)
        output = 1.
        for i in range(len(final_feature_size)):
            output *= final_feature_size[i]
        output *= self.conv_lists[-1]
        output = int(output)
        return output
    
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
        strides_len = (len(self.input_shape) - 1)
        output = x
        for conv_i in range(len(self.conv_lists)-1):
            output = self.conv3D(output, self.filters[conv_i],
                                strides = [1]*strides_len, padding = "SAME")
            output = nn.pool(output, [3]*strides_len, "MAX", [2]*strides_len, "SAME")
        output = tf.reshape(output, [output.shape[0], -1])
        output = self.mlp(output)
        return output

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
    
def main():
    # prepare dataset
    mnist = k.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = tf.cast(x_train, tf.float32), tf.cast(x_test, tf.float32)
    x_train = x_train[:, :, tf.newaxis, :, tf.newaxis]
    x_test = x_test[:, :, tf.newaxis, :, tf.newaxis]
    x_train = tf.tile(x_train, [1, 1, x_train.shape[1], 1, 1])
    x_test = tf.tile(x_test, [1, 1, x_train.shape[1], 1, 1])
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    # basic training parameters
    batch_size = 10
    epoch = 50
    lr_initial = 2e-4
    decay_steps = 1000

    # customized parameters
    input_shape = x_train.shape[1:]
    print("--------------- input shape -----------------")
    print(input_shape)
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

    # start training
    for epoch_i in tqdm(range(epoch)):
        for data, labels, total_batches, batch_id in \
                    trainDataGenerator(x_train, y_train, batch_size):
            loss = train_step(m, optimizer, data, labels)
            
            if (total_batches*epoch_i + batch_id) % 100 == 0:
                metrics.reset_states()
                for test_data, test_labels, _, _ in \
                    trainDataGenerator(x_test, y_test, batch_size):
                    test_step(m, metrics, x_test, y_test)
                acc = metrics.result()
                with summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=total_batches*epoch_i + batch_id)
                    tf.summary.scalar('accuracy', acc, step=total_batches*epoch_i + batch_id)

                save_path = ckpt_manager.save()
                print("Saved checkpoint for step {}: {}".format(\
                            int(total_batches*epoch_i+batch_id), save_path))

if __name__ == "__main__":
    main() 
