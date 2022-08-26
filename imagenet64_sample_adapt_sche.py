"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

#from improved_diffusion import dist_util, logger
import improved_diffusion.logger as logger
import time



from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)


import torchvision
import torchvision.transforms as transforms
from PIL import Image
import imageio

import torch.nn as nn



def main():
    args = create_argparser().parse_args()

    #dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    from ys_solver_pytorch import ys_solver
    my_solver = ys_solver( 
        diffusion = diffusion, 
        thres = float(args.thres),
        dpm_indices = None, 
        use_adpt = True)
    

    

    print( args.thres,  diffusion.num_timesteps, ' check num timesteps' ) 
    #model.load_state_dict(th.load( args.model_path), strict = True)
    model.load_state_dict(th.load( args.model_path))
    #device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    #model = nn.DataParallel(model)
    model.to( device )
    model.eval()

    ### 注释掉MPi相关部分
    #model.load_state_dict(
    #    dist_util.load_state_dict(args.model_path, map_location="cpu")
    #)
    #model.to(dist_util.dev())
    #model.eval()

    sum_num_s = 0 
    t_sche_list =[]
    start = time.time()
    logger.log("sampling...")
    for b_idx in range( args.num_samples // args.batch_size ):
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,),
                #device=dist_util.dev()
                device = device, 
            )
            model_kwargs["y"] = classes
       
        sample, num_s, t_idx = my_solver.sample_loop(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
        sum_num_s += num_s
       
        t_sche_list.append( t_idx )

        if b_idx == 0:
            all_images = sample
        else:
            all_images = th.cat( ( all_images, sample ), dim = 0 )
        if (b_idx + 1)% 10 == 0:
            print( time.time() - start, 'timer', b_idx + 1 )
            print( sum_num_s.item() /(b_idx + 1), 'ave_step' ) 

        logger.log(f"created { (b_idx + 1) * args.batch_size} samples")

    
    if args.npz_path != None:
        sample = (all_images + 1)* 0.5
        if args.clip_denoised == False:
            sample = th.clamp(sample, 0.0, 1.0)
        sample = (sample * 255).clamp(0, 255).to(th.uint8)
        #sample = ((all_images + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        arr = sample.cpu().numpy()
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = './results/save_npz/' + args.npz_path+ f'{shape_str}.npz'
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr)

    if args.fig_path != None:
        unloader = transforms.ToPILImage()
        imgs = 0.5 * (all_images[: 100] +1)
        if args.clip_denoised == False:
            imgs = th.clamp(imgs, 0.0, 1.0)
        imgs = torchvision.utils.make_grid(imgs, nrow =10)
        imgs = [unloader( imgs)]
        imageio.mimsave(args.fig_path, imgs)
        print('saved ' + args.fig_path ) 

    if args.schedule_path != None:
        nfe = int( args.nfe)
        for idx, t_idx in enumerate( t_sche_list ):
            if idx == 0:
                t_sche = t_idx.T[:, :nfe +1]
            else:
                t_sche = th.cat( (t_sche, t_idx.T[:, :nfe +1]), dim = 0 )
        t_sche_10 = t_sche[ t_sche[:, -1] == -1 ]
        t_sche_10 = t_sche_10[ t_sche_10[:, nfe-1] >-1 ]
        t_sche_10 = t_sche_10.float().mean(0).round().long().tolist()
        

        custom_sche = []
        for idx in range(len(t_sche_10)-1):
            custom_sche.append( [t_sche_10[idx], t_sche_10[idx +1]] )
        
        th.save( custom_sche, args.schedule_path )
        print( custom_sche, 'custom_sche' )
        print('saved ' + args.schedule_path ) 

    print( sum_num_s.item() /(b_idx + 1), 'ave_step' )

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=200,
        batch_size=50,
        use_ddim= True,
        model_path= './checkpoints/imagenet64_uncond_100M_1500K.pt',
        npz_path = None, 
        fig_path = None,
        schedule_path = None, 
        thres = None, 
        nfe = None, 
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
