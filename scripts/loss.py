from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as K
import tensorflow as tf
import math


smooth = 1.

#——————————
# Cross_entropy_loss
#——————————
def Cross_entropy_loss(y_true, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    crossEntropyLoss = -y_true * tf.log(y_pred)

    return tf.reduce_sum(crossEntropyLoss, -1)

#——————————
# dice_loss
#——————————
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)
    
    
#——————————
# focal_loss
#——————————
def binary_focal_loss(y_true, y_pred) :
    gamma = 2
    alpha = 0.25
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)

    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (K.ones_like(y_true) - y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    return K.mean(focal_loss)

#——————————
# tversky_loss
#——————————
def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1 - y_pred_pos))
    false_pos = K.sum((1 - y_true_pos) * y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)
    
    
#——————————
# c_g combo_loss
#——————————

def combo(y_true, y_pred):

    return Cross_entropy_loss(y_true, y_pred) + dice_coef_loss(y_true, y_pred)


#——————————
# c_g focal_dice
#——————————
def focal_dice(y_true, y_pred):
    t = 1 / 2

    return 1. - (dice_coef(y_true, y_pred) ** t)


#——————————
# c_g focal_tver
#——————————
def focal_tver(y_true, y_pred):
    r = 3 / 4

    return tversky_loss(y_true, y_pred) ** r


#——————————
# c_g dice_plus_focal
#——————————
def dice_plus_focal(y_true, y_pred) :
    s = 1 / 2
    return dice_coef_loss(y_true, y_pred) + s * binary_focal_loss(y_true, y_pred)


#——————————
# c_g ce_plus_energy with out dynamic weighting factor
#——————————
def ce_energy(y_true, y_pred):
    return energy_l(y_true, y_pred) + Cross_entropy_loss(y_true, y_pred)


#——————————
# multi-class function extensions
#——————————
def multi_conbination_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.
    for i in range(y_pred_n.shape[1]):
        single_loss = tversky_loss(y_true_n[:, i], y_pred_n[:, i])#Change the definition here to implement different multi-class functions
        total_single_loss += single_loss
    return total_single_loss




#——————————
# Dynamic_Energy Loss
#——————————
def area_l(true, seg):
    pos_g = K.flatten(true)
    pos_p = K.flatten(seg)
    mul_p_g = pos_g * pos_p
    area_size = K.sum(pos_g - mul_p_g) + K.sum(pos_p - mul_p_g)

    return area_size


def energy_l(true, seg):
    pos_contour = K.flatten(true)
    pos_g = K.flatten(seg)

    con_mask = K.sum(pos_contour)
    con_no_mask = tf.cast(tf.size(pos_contour), dtype=tf.float32) - con_mask

    num_no_mask = K.sum(pos_g - (pos_contour * pos_g))
    num_mask = K.sum(pos_contour * pos_g)


    pix_a_no_mask = num_no_mask / (con_no_mask + smooth)
    pix_a_mask = num_mask / (con_mask + smooth)

    energy_mask = (((tf.abs(1 - pix_a_mask)) ** 2) * num_mask) + (((tf.abs(0 - pix_a_mask)) ** 2) * tf.abs(con_mask - num_mask))
    energy_no_mask = (((tf.abs(1 - pix_a_mask)) ** 2) * num_no_mask) + (
            (tf.abs(0 - pix_a_no_mask) ** 2) * tf.abs(con_no_mask - num_no_mask))
    return energy_mask + energy_no_mask



def multi_DE_loss(y_true, y_pred):
    y_true_n = K.reshape(y_true, shape=(-1, 4))
    y_pred_n = K.reshape(y_pred, shape=(-1, 4))
    total_single_loss = 0.
    wl = 0.

    for i in range(y_pred_n.shape[1]):
        area_v = area_l(y_true_n[:, i], y_pred_n[:, i])
        alph = tf.cast(area_v, dtype=tf.float32)/tf.cast(tf.size(y_true_n[:, i]), dtype=tf.float32)
        alph = -(alph - 1) ** 4 + 10 / 7
        alph = alph * 0.7
        single_loss = (1-alph) * energy_l(y_true_n[:, i], y_pred_n[:, i]) + alph * Cross_entropy_loss(y_true_n[:, i], y_pred_n[:, i])

        total_single_loss += single_loss

    return total_single_loss

