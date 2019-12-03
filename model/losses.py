
import tensorflow as tf
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output,t='GAN'):

    if t=='GAN':
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    elif t=='WGAN' or t=='WGAN-GP':
        return -tf.reduce_mean(fake_output)
    elif t=='LSGAN':

        return tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(fake_output),fake_output))

    else:
        return cross_entropy(tf.ones_like(fake_output), fake_output) - cross_entropy(tf.zeros_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output,t='GAN'):

    if t=='GAN': 
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    elif t=='WGAN' or t=='WGAN-GP':
        return -tf.reduce_mean(real_output)+tf.reduce_mean(fake_output)

    elif t=='LSGAN':
        g_loss = tf.reduce_mean(tf.keras.losses.MSE(tf.zeros_like(fake_output),fake_output))
        r_loss = tf.reduce_mean(tf.keras.losses.MSE(tf.ones_like(real_output),real_output))
        return (g_loss+r_loss)*0.5


def discriminator_pre_loss(real_output,fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
    