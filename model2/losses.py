#coding:utf-8
'''
定义一些常用的loss function

'''
import tensorflow as tf


huber = tf.keras.losses.Huber(delta=4)


def regress_mse_loss(real,pred):

    if real.shape!=pred.shape:
        print('ERROR: NOT SAME SHAPE IN RESULT.')

    loss = tf.keras.losses.MSE(real, pred)

    return tf.reduce_mean(loss)


def regress_mae_loss(real,pred):

    if real.shape!=pred.shape:
        print('ERROR: NOT SAME SHAPE IN RESULT.')

    loss = tf.keras.losses.MAE(real, pred)

    return tf.reduce_mean(loss)


def regress_huber_loss(real,pred):
    if real.shape!=pred.shape:
        print('ERROR: NOT SAME SHAPE IN RESULT.')

    loss = huber(real, pred)

    return tf.reduce_mean(loss)


