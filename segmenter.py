import tensorflow as tf
import numpy as np
import util as util
from layers import conv2d,weight_variable,bias_variable,pixel_wise_softmax
import logging
import os
import shutil
from collections import OrderedDict
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def create_conv_net(x, keep_prob, channels, features_root = 30, layers=4, n_class=2, filter_size=3):
    

    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    convs_1 = OrderedDict()
    convs_2 = OrderedDict()
    variables = []

    #High res
    for layer in range(layers):
        with tf.name_scope("conv_pair_{}".format(str(layer))):
            features = features_root + 10*layer

            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            
            if layer == layers-1 :
                features = 2*(features-10)

            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
            elif layer == layers -1 :
                w1 = weight_variable([filter_size, filter_size, features//2, features], stddev, name="w1")

            else:
                w1 = weight_variable([filter_size, filter_size, features - 10, features], stddev, name="w1")

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            b1 = bias_variable([features], name="b1")
            b2 = bias_variable([features], name="b2")

            conv1 = conv2d(in_node, w1, b1, keep_prob)
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob)
            convs_1[layer] = tmp_h_conv
            convs_2[layer] = tf.nn.relu(conv2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            in_node = convs_2[layer]

    with tf.name_scope('output_map'):
        w_f = weight_variable([1, 1, features,n_class])
        b_f = bias_variable([n_class], name="b_f")

        conv3 = conv2d(in_node, w_f, b_f, tf.constant(1.0))
        
        output_map = tf.nn.relu(conv3)
        convs.append(output_map)
        print(np.shape(output_map))
    
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)
        
    variables.append(w_f)
    variables.append(b_f)



    return output_map,variables


class Net():

    def __init__(self, channels=3, n_class=2, cost="dice_coefficient"):
        tf.reset_default_graph()

        self.n_class = n_class

        self.x = tf.placeholder("float", shape=[None, None, None, channels], name="x")
        self.y = tf.placeholder("float", shape=[None, None, None, n_class], name="y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout (keep probability)

        logits, self.variables = create_conv_net(self.x, self.keep_prob, channels=channels, n_class=n_class)
        self.cost = self._get_cost(logits, cost)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        with tf.name_scope("results"):
            self.predicter = pixel_wise_softmax(logits)
            self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits, cost_name):
        with tf.name_scope("cost"):
            
            if cost_name == "dice_coefficient":
                eps = 1e-5
                prediction = tf.reshape(pixel_wise_softmax(logits),(2,300,300))[1]
                y_1 = tf.reshape(self.y,(2,300,300))[1]
                intersection = tf.reduce_sum(prediction * y_1)
                union = eps + tf.reduce_sum(prediction) + tf.reduce_sum(y_1)
                loss = -(2 * intersection / (union))

            else:
                raise ValueError("Unknown cost function: " % cost_name)

            return loss

    def predict(self, model_path, x_test):
        """
        Uses the model to create a prediction for the given data
        :param model_path: path to the model checkpoint to restore
        :param x_test: Data to predict on. Shape [n, nx, ny, channels]
        :returns prediction: The unet prediction Shape [n, px, py, labels] (px=nx-self.offset/2)
        """

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)

            # Restore model weights from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((x_test.shape[0], x_test.shape[1], x_test.shape[2], self.n_class))
            prediction = sess.run(self.predicter, feed_dict={self.x: x_test, self.y: y_dummy, self.keep_prob: 1.})

        return prediction

    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)



class Trainer(object):
    """
    Trains a unet instance
    :param net: the unet instance to train
    :param batch_size: size of training batch
    :param verification_batch_size: size of verification batch
    :param norm_grads: (optional) true if normalized gradients should be added to the summaries
    :param optimizer: (optional) name of the optimizer to use (momentum or adam)
    :param opt_kwargs: (optional) kwargs passed to the learning rate (momentum opt) and to the optimizer
    """

    def __init__(self, net, batch_size=1, verification_batch_size = 4, norm_grads=False, optimizer="adam", opt_kwargs={}):
        self.net = net
        self.batch_size = batch_size
        self.verification_batch_size = verification_batch_size
        self.norm_grads = norm_grads
        self.optimizer = optimizer
        self.opt_kwargs = opt_kwargs

    def _get_optimizer(self):
        if self.optimizer == "adam":
            learning_rate = 0.001
            self.learning_rate_node = tf.Variable(learning_rate, name="learning_rate")

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node).minimize(self.net.cost)
                                                                         
        return optimizer

    def _initialize(self, training_iters, output_path, restore, prediction_path):

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]), name="norm_gradients")

        
        tf.summary.scalar('loss', self.net.cost)
        tf.summary.scalar('accuracy', self.net.accuracy)

        self.optimizer = self._get_optimizer()
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        self.prediction_path = prediction_path
        abs_prediction_path = os.path.abspath(self.prediction_path)
        output_path = os.path.abspath(output_path)

        if not restore:
            logging.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            logging.info("Removing '{:}'".format(output_path))
            shutil.rmtree(output_path, ignore_errors=True)

        if not os.path.exists(abs_prediction_path):
            logging.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(output_path):
            logging.info("Allocating '{:}'".format(output_path))
            os.makedirs(output_path)

        return init

    def train(self, data_provider, output_path, training_iters=10, epochs=20, dropout=0.75, display_step=1,
              restore=False, write_graph=False, prediction_path='prediction'):
        """
        Lauches the training process
        :param data_provider: callable returning training and verification data
        :param output_path: path where to store checkpoints
        :param training_iters: number of training mini batch iteration
        :param epochs: number of epochs
        :param dropout: dropout probability
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        :param prediction_path: path where to save predictions on each epoch
        """
        save_path = os.path.join(output_path, "model.ckpt")
        if epochs == 0:
            return save_path

        init = self._initialize(training_iters, output_path, restore, prediction_path)

        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb", False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    self.net.restore(sess, ckpt.model_checkpoint_path)

            test_x, test_y = data_provider(self.verification_batch_size)
            pred_shape = [300,300,2]

            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            logging.info("Start optimization")

            avg_gradients = None
            for epoch in range(epochs):
                total_loss = 0
                for step in range((epoch * training_iters), ((epoch + 1) * training_iters)):
                    batch_x, batch_y = data_provider(self.batch_size)

                    # Run optimization op (backprop)
                    _, loss, lr, gradients = sess.run(
                        (self.optimizer, self.net.cost, self.learning_rate_node, self.net.gradients_node),
                        feed_dict={self.net.x: batch_x,
                                   self.net.y: batch_y,
                                   self.net.keep_prob: dropout})
                    if step % display_step == 0:
                        self.output_minibatch_stats(sess, summary_writer, step, batch_x, batch_y)
                    #self.store_prediction(sess,batch_x,batch_y,"epoch_%sstep_%s" % (epoch,step))
                    


                    total_loss += loss

                self.output_epoch_stats(epoch, total_loss, training_iters, lr)
                self.store_prediction(sess, test_x, test_y, "epoch_%s" % epoch)
                

                save_path = self.net.save(sess, save_path)
            logging.info("Optimization Finished!")

            return save_path

    def store_prediction(self, sess, batch_x, batch_y, name):
        prediction = sess.run(self.net.predicter, feed_dict={self.net.x: batch_x,
                                                             self.net.y: batch_y,
                                                             self.net.keep_prob: 1.})
        pred_shape = prediction.shape



        
        i=0
        for k in prediction:
            i+=1
            img = np.round(k.reshape(2,300,300)[1])
            logging.info(" mean pred {}".format(np.mean(k)))
            #util.save_image(img, "%s/%s.jpg" % (self.prediction_path, name))
            cv2.imwrite("%s/%s.jpg" % (self.prediction_path, name+str(i)),img)
        
        for k in batch_y:
            i+=1
            img = k.reshape(2,300,300)[1]
            logging.info("mean gt {}".format(np.mean(np.mean(k))))
            #util.save_image(img, "%s/%s.jpg" % (self.prediction_path, name))
            cv2.imwrite("%s/%s.jpg" % (self.prediction_path, name+"gt_"+str(i)),img)
        return pred_shape

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        logging.info(
            "Epoch {:}, Average loss: {:.4f}, learning rate: {:.4f}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op,
                                                        self.net.cost,
                                                        self.net.accuracy,
                                                        self.net.predicter],
                                                       feed_dict={self.net.x: batch_x,
                                                                  self.net.y: batch_y,
                                                                  self.net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        logging.info("Accuracy {}".format(acc))
    
    
def data_provider(number_of_image,path='data/IM_000'):
    x = []
    y = []
    random_index = [np.random.randint(899) for i in range(number_of_image)]
    for k in range(number_of_image):
        name = str(random_index[k])
        while len(name) < 3:
            name = '0'+name
        im_x = cv2.imread(path+name+'.jpg')
        im_y = cv2.imread(path+name+'_Segmentation.jpg', 0)
        im_x = cv2.resize(im_x,(300,300))
        im_y = cv2.resize(im_y,(300,300))
        x.append(im_x)
        y.append(np.array((255-im_y,im_y)).reshape(300,300,2))
    return np.array(x),np.array(y)/255
        
      
net = Net()
trainer = Trainer(net)
trainer.train(data_provider,'output/')




    
