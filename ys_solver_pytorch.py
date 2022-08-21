import math
import numpy as np
import torch as th

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)



class ys_solver:
    """
    ys_solver accelarates ddpm
    """

    def __init__(
        self,
        #model,
        diffusion = None,
        cond_fn = None,
        thres = None,
        dpm_indices = None, 
        use_adpt = True, 
        finite_diff = True, 
    ):

        #自定义变量
        #self.model = model
        self.diffusion = diffusion
        self.cond_fn = cond_fn
        self.thres = thres
        self.use_adapt = use_adpt
        self.dpm_indices = dpm_indices
        self.finite_diff = finite_diff
      
      
        self.num_timesteps = diffusion.num_timesteps
        self.alphas_cumprod = diffusion.alphas_cumprod
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        self.alphas_cumprod_all = np.append(1.0, self.alphas_cumprod)
        self.alphas_cumprod_plus = np.append(1.0 - 1e-12, self.alphas_cumprod_all)
        assert self.alphas_cumprod_prev.shape == (diffusion.num_timesteps,)


    def sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples from the model using DDIM.

        Same usage as p_sample_loop().
        """
        final = None
        for sample in self.sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final = sample
        if self.use_adapt == True:
            return final["sample"], final["num_s"], final['list']
        else:
            return final["sample"]

    
    def sample_loop_progressive(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Use DDIM to sample from the model and yield intermediate samples from
        each timestep of DDIM.

        Same usage as p_sample_loop_progressive().
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            print('Handcradting initilization')
            img = noise
        else:
            img = th.randn(*shape, device=device)
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            # Lazy import so that we don't depend on tqdm.
            from tqdm.auto import tqdm

            indices = tqdm(indices)


        if self.use_adapt == True:
            i = self.diffusion.num_timesteps - 1
            t = th.tensor([i] * shape[0], device=device)
            t_schedule = t.unsqueeze(0)
            while th.max(t) > -1:
                with th.no_grad():
                    out = self.adp_sample(
                       model,
                       img,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )
                    yield out
                img = out["sample"]
                t = out['next_step']
                t_schedule = th.cat( (t_schedule, t.unsqueeze(0)), dim =0)

            num_s = t_schedule + 0   
            num_s[ num_s > -1 ] = 1. 
            num_s[ num_s <= -1 ] = 0.  
            #print(num_s)
            out['num_s'] = (num_s.sum() / shape[0]) 
            out['list'] = t_schedule

        elif self.dpm_indices != None:
            print('dpm indices')
            indices = self.dpm_indices
            print( indices )
            for idx, i_list in enumerate(indices):
                if len(i_list) == 3:
                    i, i_mid, i_prev = i_list[0], i_list[1], i_list[2]
                    t_mid = th.tensor([i_mid] * shape[0], device=device)
                elif len(i_list) == 2:
                    i, i_prev =  i_list[0], i_list[1]
                t = th.tensor([i] * shape[0], device=device)
                t_prev = th.tensor([i_prev] * shape[0], device=device)

                if len(i_list) == 3:
                    all_t = [ t, t_mid, t_prev ]
                elif len(i_list) == 2:
                    all_t = [t, t_prev]
                with th.no_grad():
                    out = self.customize_sample(
                        model,
                        img,
                        all_t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                    )
                    yield out
                    img = out["sample"]
                
        else:
            assert 'neither choose adpt or customized'


    def adp_sample(
        self,
        model,
        x_adp,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
       
        alpha_bar = _extract_into_tensor(self.alphas_cumprod_all, t +1 , x_adp.shape)
        if self.finite_diff == True:
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_plus, t + 1 , x_adp.shape)
       
        with th.enable_grad():
            x_inputs = x_adp.detach().requires_grad_(True)
            out = self.diffusion.p_mean_variance(
                model,
                x_inputs,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                model_kwargs=model_kwargs,
            )
            if self.cond_fn is not None:
                out = self.diffusion.condition_score(self.cond_fn, out, x, t, model_kwargs=model_kwargs)
            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            eps = self.diffusion._predict_eps_from_xstart(x_inputs, t, out["pred_xstart"])

            if self.finite_diff == True:
                coef1 = 1./ ( th.sqrt( alpha_bar ) * ( th.sqrt(alpha_bar_prev) + th.sqrt( alpha_bar ) ) )
                coef2 = - 1./ ( th.sqrt( 1. - alpha_bar_prev ) + th.sqrt( 
                    1. - alpha_bar ) ) - th.sqrt( ( 1. - alpha_bar ) / alpha_bar ) / ( th.sqrt(alpha_bar_prev) + th.sqrt( alpha_bar ) )
                f_th = coef1 * x_inputs + coef2 * eps
            else:
                 ##differential formulation
                print('differential')
                f_th = 0.5 * x_inputs / alpha_bar - 0.5 * eps / ( alpha_bar * th.sqrt( 1. - alpha_bar ) )

            error = (f_th **2).sum()
            gt = th.autograd.grad(error, x_inputs)[0]
         
        with th.no_grad():
            thres = self.thres 
            gt_norm = th.sqrt( gt.pow(2).sum(3).sum(2).sum(1, keepdim= True) )
           
            stride1, stride2 = th.sqrt( 
                thres/ gt_norm ), (1. - alpha_bar.mean(3).mean(2).mean(1, keepdim= True))
            stride_alpha = th.where(stride1 < stride2, stride1, stride2)
            alpha_bar_prev_hat = 1. - stride2 + stride_alpha

            alphas_cumprod_prev_tensor = th.from_numpy( self.alphas_cumprod_prev ).unsqueeze(0).repeat( 
                alpha_bar_prev_hat.shape[0], 1 ).to( alpha_bar_prev_hat.device )
            t_prev = alphas_cumprod_prev_tensor - alpha_bar_prev_hat

            t_prev[ t_prev > 0. ] = 1
            t_prev[ t_prev <= 0. ] = 0
            t_prev = t_prev.sum(1).long()
            t_prev = th.where( t_prev < t, t_prev, t)
            t_prev = th.where( t_prev > 0, t_prev, 0)
            
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_all, t_prev, x_adp.shape)       
            coef1 = ( th.sqrt(alpha_bar_prev) - th.sqrt( alpha_bar ) ) / th.sqrt( alpha_bar )
            coef2 = th.sqrt( 1. - alpha_bar_prev ) - th.sqrt( 
                1. - alpha_bar ) - th.sqrt( (1. - alpha_bar ) / alpha_bar) * ( th.sqrt(  alpha_bar_prev ) - th.sqrt( alpha_bar ) )
            df_th = coef1 * x_inputs + coef2 * eps
            return {"sample": df_th + x_adp , "pred_xstart": out["pred_xstart"], 'next_step': t_prev-1 }   


    def customize_sample(
        self,
        model,
        x,
        all_t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using customized schedule DDIM.
        Same usage as p_sample().
        """

        
        t, t_prev = all_t[0], all_t[1]
        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        if int(t_prev[0]) == -1:
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t_prev + 1, x.shape) 
        else:
            alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod, t_prev, x.shape)
        alpha_bar_mid = alpha_bar_prev + 0. 
           
        out = self.diffusion.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        if self.cond_fn is not None:
                out = self.diffusion.condition_score(self.cond_fn, out, x, t, model_kwargs=model_kwargs)
        
        # Usually our model outputs epsilon, but we re-derive it
        # in case we used x_start or x_prev prediction.
        eps = self.diffusion._predict_eps_from_xstart(x, t, out["pred_xstart"])

        coef1 = th.sqrt(alpha_bar_mid/ alpha_bar )  
        coef2 = th.sqrt( 1. - alpha_bar_mid ) - th.sqrt( 
            1. - alpha_bar ) - th.sqrt( (1. - alpha_bar ) / alpha_bar) * ( th.sqrt(  alpha_bar_mid ) - th.sqrt( alpha_bar ) )
        x_mid = coef1 * x + coef2 * eps
        return {"sample": x_mid}

       

        
