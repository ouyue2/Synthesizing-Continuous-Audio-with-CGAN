import os
import numpy as np
import tensorflow as tf

from train import train
from inference import infer




if __name__ == '__main__':
    import argparse
    import glob
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['train', 'generate'])
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
    if args.mode == 'train':
        fps = glob.glob(os.path.join(args.data_dir, '*'))
        print('Found {} audio files in specified directory'.format(len(fps)))
        train(fps, args)
    elif args.mode == 'generate':
        infer(args)
        preview(args)
