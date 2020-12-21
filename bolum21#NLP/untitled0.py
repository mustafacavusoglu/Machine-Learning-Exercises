# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:03:46 2020

@author: mustdo
"""


import tensorflow as tf

hello = tf.constant("hello tensorflow")
sess = tf.Session()
print(sess.run(hello))