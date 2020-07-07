import os
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from glob import glob
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2DTranspose
from tensorflow.keras.layers import Input, Dropout, Conv2D, MaxPooling2D


class U_Net():
    def __init__(self):
        # 图片基本参数
        self.height = 256
        self.width = 256
        self.channels = 1
        self.shape = (self.height, self.width, self.channels)

        # 优化器
        self.optimizer = Adam(0.0002, 0.5)
        # self.optimizer = Adam(0.0002, 0.5)

        #u_net
        self.unet = self.build_unet()
        # self.unet.compile(loss=self.loss_fun,
        #                   optimizer=optimizer,
        #                   metrics=['accuracy'])
        self.unet.summary()


    def build_unet(self, n_filters=16, dropout=0, batchnorm=True, padding='same'):

        # 定义一个多次使用的卷积块
        def conv2d_block(input_tensor, n_filters=16, kernel_size=3, batchnorm=True, padding='same'):
            # the first layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(
                input_tensor)
            if batchnorm:
                x = BatchNormalization()(x)
            x = Activation('relu')(x)

            # the second layer
            x = Conv2D(n_filters, kernel_size, padding=padding)(x)
            if batchnorm:
                x = BatchNormalization()(x)
            X = Activation('relu')(x)
            return X

        # 构建一个输入
        img = Input(shape=self.shape)

        # contracting path
        c1 = conv2d_block(img, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout * 0.5)(p1)

        c2 = conv2d_block(p1, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = conv2d_block(p2, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = conv2d_block(p3, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = conv2d_block(p4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm, padding=padding)

        # extending path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = conv2d_block(u6, n_filters=n_filters * 8, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = conv2d_block(u7, n_filters=n_filters * 4, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = conv2d_block(u8, n_filters=n_filters * 2, kernel_size=3, batchnorm=batchnorm, padding=padding)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = conv2d_block(u9, n_filters=n_filters * 1, kernel_size=3, batchnorm=batchnorm, padding=padding)

        output = Conv2D(1, (1, 1), activation='sigmoid')(c9)

        return Model(img, output)

    def loss_fun(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))   # 返回mse

    # def acc_fun(self, A, B):
    #     batch_size = A.shape[0]
    #     metric = []
    #     for batch in range(batch_size):
    #         t, p = A[batch], tf.where(B[batch]>0.1,1,0)
    #         print(p)
    #         intersection = np.logical_and(t,p)
    #         union = np.logical_or(t, p)
    #         iou = (np.sum(intersection>0)+1e-10)/(np.sum(union>0)+1e-10)
    #         thresholds = np.arange(0.5, 1, 0.05)
    #         s = []
    #         for thresh in thresholds:
    #             s.append(iou>thresh)
    #         metric.append(np.mean(s))
    #     return np.mean(metric)

    def acc_fun(self, y_true, y_pred):
        batch_size = y_true.shape[0]
        metric = []
        for batch in range(batch_size):
            intersection = y_true[batch]*tf.cast(y_pred[batch]>0.1, dtype=tf.float32)
            union = (y_true[batch]+tf.cast(y_pred[batch]>0.1, dtype=tf.float32))-intersection
            iou = (tf.reduce_sum(intersection)+1e-10)/(tf.reduce_sum(union)+1e-10)
            metric.append(iou)
        return tf.reduce_mean(metric)

    def load_data(self):
        x_train = []
        x_label = []
        for file in glob('./train/*'):
            for filename in glob(file+'/*'):
                img = np.array(Image.open(filename), dtype='float32')/255
                x_train.append(img[256:, 128:384])
        for file in glob('./label/*'):
            for filename in glob(file+'/*'):
                img = np.array(Image.open(filename), dtype='float32')/255
                x_label.append(img[256:, 128:384])
        return np.array(x_train), np.array(x_label)

    def train(self, epochs=51, batch_size=32, sample_interval=10):
        os.makedirs('./weights', exist_ok=True)
        # 获得数据
        x_train, x_label = self.load_data()
        np.random.seed(116)
        np.random.shuffle(x_train)
        np.random.seed(116)
        np.random.shuffle(x_label)
        x_train = tf.constant(np.expand_dims(x_train, axis=3), dtype=tf.float32)
        x_label = tf.constant(np.expand_dims(x_label, axis=3), dtype=tf.float32)
        total_num = x_train.shape[0]
        train_num = total_num-300
        print(total_num)

        for epoch in range(epochs):
            loss, acc = [], []
            train_loss, train_acc = 0, 0
            index, step = 0, 0

            while index < train_num:
                step += 1   # 记录训练批数
                with tf.GradientTape() as tape:
                    # 前向传播
                    img = self.unet(x_train[index:index+batch_size])

                    # 计算loss
                    step_loss = self.loss_fun(x_label[index:index+batch_size], img)
                    step_acc = 0.78 # self.acc_fun(x_label[index:index+batch_size], img)
                

                # 计算梯度并更新权重
                grads = tape.gradient(step_loss, self.unet.trainable_weights)
                self.optimizer.apply_gradients(zip(grads, self.unet.trainable_weights))
                train_loss += step_loss
                train_acc += step_acc
                # print(step_loss.shape)
                print('schedule: %d/%d' % (index, train_num),
                      ' - loss: %f, - acc: %f%%' % (step_loss, step_acc * 100))
                index += batch_size

            # for step, (x_train, x_label) in enumerate(train_db):
            #     step_loss, step_acc = self.unet.train_on_batch(x_train, x_label)
            #     train_loss += step_loss
            #     train_acc += step_acc
            #     print('schedule: %.2f%%' % step/x_train.shape[0],
            #           ' - loss: %f, - acc: %.2f%%' % (step_loss, step_acc*100))

            train_loss /= step
            train_acc /= step
            loss.append(train_loss)
            acc.append(train_acc)
            print('[Epoch %d/%d] [D loss: %f, acc: %3d%%]'
                  % (epoch, epochs, train_loss, train_acc*100))
            if epoch % sample_interval == 0:
                self.unet.save_weights('./weights/unet_epoch%d.h5' % epoch)

        # results = self.unet.fit(x_train, x_label, batch_size=32, epochs=200, verbose=1)

        plt.figure(figsize=(8, 8))
        plt.plot(loss, label='loss')
        plt.title("Learning curve")
        plt.xlabel("Epochs")
        plt.ylabel("log_loss")
        plt.show()
    def test(self, batch_size=1):
        os.makedirs('./test_result', exist_ok=True)
        self.unet.load_weights(r"weights/unet_epoch50.h5")
        # 获得数据
        x_train, x_label = self.load_data()
        test_num = 300
        x_train = np.expand_dims(x_train[-test_num:], axis=3)
        x_label = np.expand_dims(x_label[-test_num:], axis=3)
        index, step = 0, 0

        while index < test_num:
            step += 1  # 记录训练批数
            mask = self.unet.predict(x_train[index:index + batch_size])
            print('schedule: %d/%d' % (index, test_num))
            mask = Image.fromarray(np.uint8((mask[0,:,:,0]>0.1)*255))
            mask.save('./test_result/'+str(step)+'.png')
            mask_true = Image.fromarray(np.uint8(x_label[index,:,:,0]*255))
            mask_true.save('./test_result/'+str(step)+'true.png')
            index += batch_size


if __name__ == '__main__':
    unet = U_Net()
    unet.test()







