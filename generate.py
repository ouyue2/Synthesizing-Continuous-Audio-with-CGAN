import os
import numpy as np
import tensorflow as tf

from inference import infer

def generate(args):
    import librosa
    
    infer_dir = os.path.join(args.train_dir, 'infer')
    infer_metagraph_fp = os.path.join(infer_dir, 'infer.meta')
    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(infer_metagraph_fp)
    graph = tf.get_default_graph()
    
    with tf.Session() as sess:
        saver.restore(sess, args.checkpoint)
        z = graph.get_tensor_by_name('z:0')
        y = graph.get_tensor_by_name('y:0')
        G_z = graph.get_tensor_by_name('G_z:0')[:, :, 0]
        
        # Loop_Init
        _y = np.zeros([1, args.wavegan_genr_pp_len,1])
        wv = np.zeros([1,1])
        gen_count = 0
        
        # Loop
        while True:
            _z = (np.random.rand(1, 100) * 2.) - 1.
            wv = np.concatenate((wv,sess.run(G_z, {y: _y, z: _z})), axis = 1)
            _y = np.reshape(wv[:,-1-args.wavegan_genr_pp_len:-1], (1,-1,1))
            gen_count = gen_count+1
            
            if gen_count==4:
                librosa.output.write_wav(args.wav_out_path, wv[0, :], 16000)
                gen_count=0


if __name__ == '__main__':
    import argparse
    import glob
    import sys
	
    parser = argparse.ArgumentParser(description='WaveGan generation script')
	
    parser.add_argument('checkpoint', type=str, help='Which model checkpoint to generate from e.g. "(fullpath)/model.ckpt-XXX"')
    parser.add_argument('--train_dir', type=str, help='Training directory')
    parser.add_argument('--wav_out_path', type=str, help='Path to output wav file')
    parser.add_argument('--wavegan_genr_pp_len', type=int,help='Length of post-processing filter for DCGAN')
    parser.add_argument('--wavegan_latent_dim', type=int,help='Number of dimensions of the latent space')
    parser.add_argument('--data_slice_len', type=int, choices=[16384, 32768, 65536],help='Number of audio samples per slice (maximum generation length)')
    parser.add_argument('--wavegan_kernel_len', type=int,help='Length of 1D filter kernels')
    parser.add_argument('--wavegan_dim', type=int,help='Dimensionality multiplier for model of G and D')
    parser.add_argument('--wavegan_batchnorm', action='store_true', dest='wavegan_batchnorm',help='Enable batchnorm')
	
    parser.set_defaults(checkpoint=None, 
                        train_dir='./train',   
                        wav_out_path='./gen.wav', 
                        wavegan_genr_pp_len=512,
                        data_slice_len=32768,
                        wavegan_latent_dim=100,
                        wavegan_kernel_len=25,
                        wavegan_dim=64,
                        wavegan_batchnorm=False)
	
    args = parser.parse_args()
    
    setattr(args, 'wavegan_g_kwargs', {
        'slice_len': args.data_slice_len,
        'kernel_len': args.wavegan_kernel_len,
        'dim': args.wavegan_dim,
        'use_batchnorm': args.wavegan_batchnorm,
    })
	
    infer(args)
    generate(args)
