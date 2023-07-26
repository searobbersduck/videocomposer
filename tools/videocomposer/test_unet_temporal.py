import os
import os.path as osp
import sys
# append parent path to environment
sys.path.insert(0, '/'.join(osp.realpath(__file__).split('/')[:-4]))
cur_dir = os.path.dirname(__file__)
print(cur_dir)
sys.path.append(cur_dir)

print('/'.join(osp.realpath(__file__).split('/')[:-4]))

import torch
import numpy as np
import random


from .config import cfg
from .unet_sd import UNetSD_temporal

import logging

from copy import deepcopy, copy
import random
import json
import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.cuda.amp as amp
import torchvision.transforms as T
import pynvml
import torchvision.transforms.functional as TF
from importlib import reload
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
import open_clip
from easydict import EasyDict
from collections import defaultdict
from functools import partial
from io import BytesIO
from PIL import Image


import artist.ops as ops
import artist.data as data


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     

def inference_single(cfg_update, **kwargs):
    cfg.update(**kwargs)

    # Copy update input parameter to current task
    for k, v in cfg_update.items():
        cfg[k] = v

    cfg.read_image = getattr(cfg, 'read_image', False)
    cfg.read_sketch = getattr(cfg, 'read_sketch', False)
    cfg.read_style = getattr(cfg, 'read_style', False)
    cfg.save_origin_video = getattr(cfg, 'save_origin_video', True)

    if not 'MASTER_ADDR' in os.environ:
        os.environ['MASTER_ADDR']='localhost'
        os.environ['MASTER_PORT']= find_free_port()
    cfg.pmi_rank = int(os.getenv('RANK', 0)) # 0
    cfg.pmi_world_size = int(os.getenv('WORLD_SIZE', 1))
    setup_seed(cfg.seed)

    if cfg.debug:
        cfg.gpus_per_machine = 1
        cfg.world_size = 1
    else:
        cfg.gpus_per_machine = torch.cuda.device_count()
        cfg.world_size = cfg.pmi_world_size * cfg.gpus_per_machine
    
    if cfg.world_size == 1:
        worker(0, cfg)
    else:
        mp.spawn(worker, nprocs=cfg.gpus_per_machine, args=(cfg, ))
    return cfg

def worker(gpu, cfg):
    cfg.gpu = gpu
    cfg.rank = cfg.pmi_rank * cfg.gpus_per_machine + gpu

    # init distributed processes
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    if not cfg.debug:
        dist.init_process_group(backend='nccl', world_size=cfg.world_size, rank=cfg.rank)

    # logging
    log_dir = ops.generalized_all_gather(cfg.log_dir)[0]
    exp_name = os.path.basename(cfg.cfg_file).split('.')[0] + '-S%05d' % (cfg.seed)
    log_dir = os.path.join(log_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    cfg.log_dir = log_dir
    if cfg.rank == 0:
        name = osp.basename(cfg.log_dir)
        cfg.log_file = osp.join(cfg.log_dir, '{}_rank{}.log'.format(name, cfg.rank))
        reload(logging)
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler(filename=cfg.log_file),
                logging.StreamHandler(stream=sys.stdout)])
        logging.info(cfg)

    # rank-wise params
    l1 = len(cfg.frame_lens)
    l2 = len(cfg.feature_framerates)
    cfg.max_frames = cfg.frame_lens[cfg.rank % (l1*l2)// l2]
    cfg.batch_size = cfg.batch_sizes[str(cfg.max_frames)]

    # zero_y.shape: [1, 77, 1024]
    # zero_y.dtype: torch.float32
    
    # black_image_feature: [1, 1, 1024]
    # black_image_feature: torch.float32
    

    zero_y = torch.randn([1, 77, 1024], dtype=torch.float32)
    black_image_feature = torch.randn([1,1,1024], dtype=torch.float32)

    model = UNetSD_temporal(
        cfg=cfg,
        in_dim=cfg.unet_in_dim,
        concat_dim= cfg.unet_concat_dim,
        dim=cfg.unet_dim,
        y_dim=cfg.unet_y_dim,
        context_dim=cfg.unet_context_dim,
        out_dim=cfg.unet_out_dim,
        dim_mult=cfg.unet_dim_mult,
        num_heads=cfg.unet_num_heads,
        head_dim=cfg.unet_head_dim,
        num_res_blocks=cfg.unet_res_blocks,
        attn_scales=cfg.unet_attn_scales,
        dropout=cfg.unet_dropout,
        temporal_attention = cfg.temporal_attention,
        temporal_attn_times = cfg.temporal_attn_times,
        use_checkpoint=cfg.use_checkpoint,
        use_fps_condition=cfg.use_fps_condition,
        use_sim_mask=cfg.use_sim_mask,
        video_compositions=cfg.video_compositions,
        misc_dropout=cfg.misc_dropout,
        p_all_zero=cfg.p_all_zero,
        p_all_keep=cfg.p_all_zero,
        zero_y = zero_y,
        black_image_feature = black_image_feature,
        )
    
    
    print('\n\n\n\nunetsd_temporal model:\n')
    print(model)
    print('\n\n\n\n')
    
    # model.load_state_dict(torch.load(DOWNLOAD_TO_CACHE(cfg.resume_checkpoint), map_location='cpu'),strict=False)
    
    '''
        xt.shape: [1, 4, 16, 32, 32]
        model_kwargs: 
            len(model_kwargs): 2
            model_kwargs[0].keys(): dict_keys(['y', 'local_image', 'motion'])
            model_kwargs[1].keys(): dict_keys(['y', 'local_image', 'motion'])
            model_kwargs[0]['y'].shape: [1, 77, 1024]
            model_kwargs[0]['y'].dtype: torch.float32
            model_kwargs[0]['local_image'].shape: [1, 3, 16, 384, 384]
            model_kwargs[0]['local_image'].dtype: torch.float32
            model_kwargs[0]['motion'].shape: [1, 2, 16, 256, 256]
            model_kwargs[0]['motion'].dtype: torch.float32
    ''' 
    
    '''
        if guide_scale is None:
            out = model(xt, self._scale_timesteps(t), **model_kwargs)
        else:
            # classifier-free guidance
            # (model_kwargs[0]: conditional kwargs; model_kwargs[1]: non-conditional kwargs)
            assert isinstance(model_kwargs, list) and len(model_kwargs) == 2
            y_out = model(xt, self._scale_timesteps(t), **model_kwargs[0])
            u_out = model(xt, self._scale_timesteps(t), **model_kwargs[1])
            dim = y_out.size(1) if self.var_type.startswith('fixed') else y_out.size(1) // 2
            out = torch.cat([
                u_out[:, :dim] + guide_scale * (y_out[:, :dim] - u_out[:, :dim]),
                y_out[:, dim:]], dim=1) # guide_scale=9.0 
    '''
    
    '''
        import torch
        import torchvision

        dummy_input = torch.randn(10, 3, 224, 224, device="cuda")
        model = torchvision.models.alexnet(pretrained=True).cuda()

        # Providing input and output names sets the display names for values
        # within the model's graph. Setting these does not change the semantics
        # of the graph; it is only for readability.
        #
        # The inputs to the network consist of the flat list of inputs (i.e.
        # the values you would pass to the forward() method) followed by the
        # flat list of parameters. You can partially specify names, i.e. provide
        # a list here shorter than the number of inputs to the model, and we will
        # only set that subset of names, starting from the beginning.
        input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
        output_names = [ "output1" ]

        torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
    
    '''

    dummy_inputs = {
        "x": torch.rand([1,4,16,32,32], dtype=torch.float32), 
        "t": torch.tensor([1], dtype=torch.int64), 
        "y": torch.rand([1, 77, 1024], dtype=torch.float32)
    }
    
    output_names = ['noise_pred']
    
    onnx_dir = os.path.join(cur_dir, 'onnx')
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, 'unetsd_temporal18.onnx')
    
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model, 
            tuple(dummy_inputs.values()), 
            onnx_path, 
            export_params=True, 
            verbose=True, 
            # opset_version=18, 
            do_constant_folding=True, 
            input_names=list(dummy_inputs.keys()), 
            output_names=output_names
        )
    
    print('congratulations!')
    

if __name__ == '__main__':
    print('hello world!')