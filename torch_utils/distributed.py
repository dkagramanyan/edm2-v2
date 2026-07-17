# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

import os
import socket

import torch
import torch.distributed

from . import training_stats

_sync_device = None

#----------------------------------------------------------------------------

def init():
    global _sync_device

    if not torch.distributed.is_initialized():
        # Setup some reasonable defaults for env-based distributed init if
        # not set by the running environment.
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(('', 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            os.environ['MASTER_PORT'] = str(s.getsockname()[1])
            s.close()
        if 'RANK' not in os.environ:
            os.environ['RANK'] = '0'
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = '0'
        if 'WORLD_SIZE' not in os.environ:
            os.environ['WORLD_SIZE'] = '1'
        backend = 'gloo' if os.name == 'nt' else 'nccl'
        init_method = 'env://?use_libuv=False' if os.name == 'nt' else 'env://'
        torch.distributed.init_process_group(backend=backend, init_method=init_method)
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))

    _sync_device = torch.device('cuda') if get_world_size() > 1 else None
    training_stats.init_multiprocessing(rank=get_rank(), sync_device=_sync_device)

#----------------------------------------------------------------------------

def get_rank():
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0

#----------------------------------------------------------------------------

def get_world_size():
    return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

#----------------------------------------------------------------------------

def print0(*args, **kwargs):
    if get_rank() == 0:
        print(*args, **kwargs)

#----------------------------------------------------------------------------
