import os
import numpy as np
import tensorflow as tf

from functools import reduce
from six.moves import xrange

import loader
from model import WaveGANGenerator, WaveGANDiscriminator


"""
  Trains a WaveGAN
"""


def train(fps, args):
    with tf.name_scope('loader'):
        x = loader.decode_extract_and_batch(fps,
                                          batch_size=args.train_batch_size,
                                          slice_len=args.data_slice_len,
                                          decode_fs=args.data_sample_rate,
                                          decode_num_channels=args.data_num_channels,
                                          decode_fast_wav=args.data_fast_wav,
                                          decode_parallel_calls=4,
                                          slice_randomize_offset=False,
                                          slice_first_only=args.data_first_slice,
                                          slice_overlap_ratio=0.,
                                          slice_pad_end=True,
                                          repeat=True,
                                          shuffle=True,
                                          shuffle_buffer_size=4096,
                                          prefetch_size=args.train_batch_size * 4,
                                          prefetch_gpu_num=args.data_prefetch_gpu_num)
        x = x[:, :, 0]

    # Make z vector
    z = tf.random_uniform([args.train_batch_size, args.wavegan_latent_dim], -1., 1., dtype=tf.float32)

    # Make generator
    with tf.variable_scope('G'):
        # use first 512 point from real data as y
        y = tf.slice(x, [0, 0, 0], [-1, args.wavegan_smooth_len, -1])
        G_z = WaveGANGenerator(y, z, train=True, y_len=args.wavegan_smooth_len, **args.wavegan_g_kwargs)
    G_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G')
    # Print G summary
    print('-' * 80)
    print('Generator vars')
    nparams = 0
    for v in G_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(),v_n,v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))

    # Make real discriminator
    with tf.name_scope('D_x'), tf.variable_scope('D'):
        D_x = WaveGANDiscriminator(x, **args.wavegan_d_kwargs)
    D_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D')

    # Print D summary
    print('-' * 80)
    print('Discriminator vars')
    nparams = 0
    for v in D_vars:
        v_shape = v.get_shape().as_list()
        v_n = reduce(lambda x, y: x * y, v_shape)
        nparams += v_n
        print('{} ({}): {}'.format(v.get_shape().as_list(),v_n,v.name))
    print('Total params: {} ({:.2f} MB)'.format(nparams, (float(nparams) * 4) / (1024 * 1024)))
    print('-' * 80)

    # Make fake discriminator
    with tf.name_scope('D_G_z'), tf.variable_scope('D', reuse=True):
        yG_z = tf.concat([y, G_z], 1)
        print("yG_z shape:")
        print(yG_z.get_shape())
        D_G_z = WaveGANDiscriminator(yG_z, **args.wavegan_d_kwargs)

    # Create loss
    D_clip_weights = None
    if args.wavegan_loss == 'dcgan':
        fake = tf.zeros([args.train_batch_size], dtype=tf.float32)
        real = tf.ones([args.train_batch_size], dtype=tf.float32)
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=real))
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_G_z, labels=fake))
        D_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_x, labels=real))
        D_loss /= 2.

    elif args.wavegan_loss == 'lsgan':
        G_loss = tf.reduce_mean((D_G_z - 1.) ** 2)
        D_loss = tf.reduce_mean((D_x - 1.) ** 2)
        D_loss += tf.reduce_mean(D_G_z ** 2)
        D_loss /= 2.
    elif args.wavegan_loss == 'wgan-gp':
        G_loss = -tf.reduce_mean(D_G_z)
        D_loss = tf.reduce_mean(D_G_z) - tf.reduce_mean(D_x)

        alpha = tf.random_uniform(shape=[args.train_batch_size, 1, 1], minval=0., maxval=1.)
        differences = yG_z - x
        interpolates = x + (alpha * differences)
        with tf.name_scope('D_interp'), tf.variable_scope('D', reuse=True):
            D_interp = WaveGANDiscriminator(interpolates, **args.wavegan_d_kwargs)
        LAMBDA = 10
        gradients = tf.gradients(D_interp, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2.)
        D_loss += LAMBDA * gradient_penalty
    else:
        raise NotImplementedError()

    # Create (recommended) optimizer
    if args.wavegan_loss == 'dcgan':
        G_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
        D_opt = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5)
    elif args.wavegan_loss == 'lsgan':
        G_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
        D_opt = tf.train.RMSPropOptimizer(learning_rate=1e-4)
    elif args.wavegan_loss == 'wgan-gp':
        G_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
        D_opt = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9)
    else:
        raise NotImplementedError()
    # Create training ops
    G_train_op = G_opt.minimize(G_loss, var_list=G_vars, global_step=tf.train.get_or_create_global_step())
    D_train_op = D_opt.minimize(D_loss, var_list=D_vars)

    # Summarize
    tf.summary.audio('x', x, args.data_sample_rate)
    tf.summary.audio('G_z', G_z, args.data_sample_rate)
    tf.summary.audio('yG_z', yG_z, args.data_sample_rate)

    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('D_loss', D_loss)

    # Run training
    with tf.train.MonitoredTrainingSession(checkpoint_dir=args.train_dir,
                                         save_checkpoint_secs=args.train_save_secs,
                                         save_summaries_secs=args.train_summary_secs) as sess:
        while True:
            # Train discriminator
            for i in xrange(args.wavegan_disc_nupdates):
                sess.run(D_train_op)

            # Enforce Lipschitz constraint for WGAN
            if D_clip_weights is not None:
                sess.run(D_clip_weights)

            # Train generator
            sess.run(G_train_op)
            if args.verbose:
                eval_loss_D = D_loss.eval(session=sess)
                eval_loss_G = G_loss.eval(session=sess)
                print(str(eval_loss_D)+","+str(eval_loss_G))




if __name__ == '__main__':
    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str,help='Training directory')
    data_args = parser.add_argument_group('Data')
    data_args.add_argument('--data_dir', type=str,help='Data directory containing *only* audio files to load')
    data_args.add_argument('--data_sample_rate', type=int,help='Number of audio samples per second')
    data_args.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],help='Number of audio samples per slice (maximum generation length)')
    data_args.add_argument('--data_num_channels', type=int,help='Number of audio channels to generate (for >2, must match that of data)')
    data_args.add_argument('--data_overlap_ratio', type=float,help='Overlap ratio [0, 1) between slices')
    data_args.add_argument('--data_first_slice', action='store_true', dest='data_first_slice',help='If set, only use the first slice each audio example')
    data_args.add_argument('--data_pad_end', action='store_true', dest='data_pad_end',help='If set, use zero-padded partial slices from the end of each audio file')
    data_args.add_argument('--data_normalize', action='store_true', dest='data_normalize',help='If set, normalize the training examples')
    data_args.add_argument('--data_fast_wav', action='store_true', dest='data_fast_wav',help='If your data is comprised of standard WAV files (16-bit signed PCM or 32-bit float), use this flag to decode audio using scipy (faster) instead of librosa')
    data_args.add_argument('--data_prefetch_gpu_num', type=int,help='If nonnegative, prefetch examples to this GPU (Tensorflow device num)')

    wavegan_args = parser.add_argument_group('WaveGAN')
    wavegan_args.add_argument('--wavegan_latent_dim', type=int,help='Number of dimensions of the latent space')
    wavegan_args.add_argument('--wavegan_kernel_len', type=int,help='Length of 1D filter kernels')
    wavegan_args.add_argument('--wavegan_dim', type=int,help='Dimensionality multiplier for model of G and D')
    wavegan_args.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',help='Enable batchnorm')
    wavegan_args.add_argument('--wavegan_disc_nupdates', type=int,help='Number of discriminator updates per generator update')
    wavegan_args.add_argument('--wavegan_loss', type=str, choices=['dcgan', 'lsgan', 'wgan', 'wgan-gp'],help='Which GAN loss to use')
    wavegan_args.add_argument('--wavegan_genr_pp_len', type=int,help='Length of post-processing filter for DCGAN')
    wavegan_args.add_argument('--wavegan_disc_phaseshuffle', type=int,help='Radius of phase shuffle operation')
    wavegan_args.add_argument('--wavegan_smooth_len', type=int, help='Length of the pervious audio used to smooth the connection')

    train_args = parser.add_argument_group('Train')
    train_args.add_argument('--train_batch_size', type=int,help='Batch size')
    train_args.add_argument('--train_save_secs', type=int,help='How often to save model')
    train_args.add_argument('--train_summary_secs', type=int,help='How often to report summaries')
    train_args.add_argument('--verbose', action='store_true',dest='verbose',help='If yes, print G and D loss to stdout')



    parser.set_defaults(train_dir = './train',
                        data_dir=None,
                        data_sample_rate=11025,
                        data_slice_len=32768,
                        data_num_channels=1,
                        data_overlap_ratio=0.,
                        data_first_slice=False,
                        data_pad_end=False,
                        data_normalize=False,
                        data_fast_wav=False,
                        data_prefetch_gpu_num=0,
                        wavegan_latent_dim=100,
                        wavegan_kernel_len=25,
                        wavegan_dim=64,
                        wavegan_batchnorm=False,
                        wavegan_disc_nupdates=5,
                        wavegan_loss='wgan-gp',
                        wavegan_genr_pp_len=512,
                        wavegan_disc_phaseshuffle=2,
                        wavegan_smooth_len=4096, 
                        train_batch_size=64,
                        train_save_secs=300,
                        train_summary_secs=120,
                        verbose=False)

    args = parser.parse_args()
    # Make model kwarg dicts
    setattr(args, 'wavegan_g_kwargs', {
        'slice_len': args.data_slice_len,
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
    })
    setattr(args, 'wavegan_d_kwargs', {
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
        'phaseshuffle_rad': args.wavegan_disc_phaseshuffle
    })
    # Make train dir
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)

    fps = glob.glob(os.path.join(args.data_dir, '*'))
       
    if len(fps) == 0:
       raise Exception('Did not find any audio files in specified directory')
    print('Found {} audio files in specified directory'.format(len(fps)))
    # infer(args)
    train(fps, args)
