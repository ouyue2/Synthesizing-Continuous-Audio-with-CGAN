import os
import numpy as np
import tensorflow as tf


def infer(args):
    from model import WaveGANGenerator
    
    infer_dir = os.path.join(args.train_dir, 'infer')
    if not os.path.isdir(infer_dir):
        os.makedirs(infer_dir)
    # Input zo
    z = tf.placeholder(tf.float32, [None, args.wavegan_latent_dim], name='z')
    y = tf.placeholder(tf.float32, [None, args.wavegan_genr_pp_len,1], name='y')

    # Execute generator
    with tf.variable_scope('G'):
        G_z = WaveGANGenerator(y,z, train=False, **args.wavegan_g_kwargs)
    G_z = tf.identity(G_z, name='G_z')

    # Create saver
    G_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='G')
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(G_vars + [global_step])

    # Export graph
    tf.train.write_graph(tf.get_default_graph(), infer_dir, 'infer.pbtxt')

    # Export MetaGraph
    infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
    tf.train.export_meta_graph(filename=infer_metagraph_fp,clear_devices=True,saver_def=saver.as_saver_def())

    # Reset graph (in case training afterwards)
    tf.reset_default_graph()


if __name__ == '__main__':
    import argparse
    import glob
    import sys
	
    parser = argparse.ArgumentParser(description='WaveGan generation script')
	
    parser.add_argument('--train_dir', type=str, help='Training directory')
    parser.add_argument('--wavegan_latent_dim', type=int,help='Number of dimensions of the latent space')
    parser.add_argument('--wavegan_genr_pp_len', type=int,help='Length of post-processing filter for DCGAN')
    parser.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],help='Number of audio samples per slice (maximum generation length)')
    parser.add_argument('--wavegan_kernel_len', type=int,help='Length of 1D filter kernels')
    parser.add_argument('--wavegan_dim', type=int,help='Dimensionality multiplier for model of G and D')
    parser.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',help='Enable batchnorm')
	
    parser.set_defaults(train_dir = './train',
                        data_slice_len=32768,
                        wavegan_latent_dim=100,
                        wavegan_kernel_len=25,
                        wavegan_dim=64,
                        wavegan_batchnorm=False,
                        wavegan_genr_pp_len=512)
	
    args = parser.parse_args()
    
    setattr(args, 'wavegan_g_kwargs', {
        'slice_len': args.data_slice_len,
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
    })
    
    infer(args)
    
