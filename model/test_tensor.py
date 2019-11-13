#coding:utf-8

### 测试TensorFlow
import tensorflow as tf

def func_y(x,z):

    return x*x+z

def test_gradient():

    x = 2
    z = 1

    y = func_y
    

    with tf.GradientTape as t:
        k = x-1
        t.watch(k)
        y = func_y(k,z)

    grad = tf.gradient(y,t)


if __name__ == '__main__':
    test_gradient()



