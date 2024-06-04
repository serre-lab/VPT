import torch
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Union, List
import argparse
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ToTensor
from timm.data.transforms import str_to_interp_mode
from submodule.probe3d.evals.models.dino import DINO
from submodule.probe3d.evals.models.sam import SAM
from submodule.probe3d.evals.models.midas_final import make_beit_backbone
from submodule.probe3d.evals.models.mae import MAE
from submodule.probe3d.evals.models.ibot import iBOT
from submodule.Depth_Anything.depth_anything.dpt import DepthAnythingEncoder, DepthAnything
from submodule.Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2


def get_args_parser():
    parser = argparse.ArgumentParser("Linear Probing with Perception Data", allow_abbrev=False)
    parser.add_argument('--data_dir', required=True, type=str)
    parser.add_argument("--best_model", required=False, type=str,
                        default = 'best_acc', choices = ['best_acc'])
    parser.add_argument('--model_name', required=False, type=str, default = 'vit_base_patch16_224.augreg_in21k_ft_in1k', help='TIMM model name')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--extract_batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument("--momentum", required=False, type = float, default = 0.9,
                        help="SGD momentum")
    parser.add_argument("--step_size", required=False, type = int, default = 25,
                        help="learning rate scheduler")
    parser.add_argument("--gamma", required=False, type = float, default = 0.1,
                        help="scheduler parameters, which decides the change of learning rate ")
    parser.add_argument("--weight_decay", required=False, type = float, default = 1e-4,
                        help="weight decay, regularization")
    parser.add_argument("--interval", required=False, type = int, default = 2,
                        help="Step interval for printing logs")
    parser.add_argument("--num_workers", required=False, type = int, default = 4,
                        help="number of workers in dataloader")
    parser.add_argument("--gpu_id", required=False, type = int, default = 0,
                        help="specify gpu id for single gpu training")
    parser.add_argument("--wandb", action='store_true', default = False,
                        help="Whether to W&B to record progress")
    parser.add_argument("--warmup", required=False, type = int, default = 3,
                        help="specify warmup epochs, usually <= 5")
    parser.add_argument("--num_classes", required=False, type = int, default = 1, help="specify number of classes of the classification head")
    parser.add_argument("--dropout_rate", required=False, type = float, default=0.3, help="Dropout rate for TIMM model")
    parser.add_argument("--task", required=False, default='perspective', choices=['perspective', 'depth'])
    parser.add_argument("--min_lr", required=False, default=1e-6, type=float, help="Minimum learning rate")
    parser.add_argument("--dpt_encoder", required=False, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument("--not_pretrained", default=False, action="store_true")
    parser.add_argument("--flip", default=False, action="store_true")
    parser.add_argument("--exp", default=False, action="store_true")
    parser.add_argument('--exp_dir', required=False, type=str)
    parser.add_argument("--ckpt_path", type=str)
    return parser

def get_transform_wo_crop(data_config):
    input_size = data_config['input_size']
    if isinstance(input_size, (tuple, list)):
        input_size = input_size[-2:]
    else:
        input_size = (input_size, input_size)
    mean = data_config['mean']
    std = data_config['std']
    interpolation = data_config['interpolation']
    tf = []
    #if input_size[0] == input_size[1]:
    tf += [transforms.Resize(input_size[0], interpolation=str_to_interp_mode(interpolation))]
    #else:
    #    tf += [ResizeKeepRatio(input_size)]
    tf += [transforms.ToTensor(), transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))]
    return transforms.Compose(tf)
    
def binary_accuracy(output, target):
    with torch.no_grad():
        output = output > 0
        return torch.sum(output==target).item()/len(target)*100
    
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
        
def build_depth_anything(args):
    device = torch.device(f'cuda:{args.gpu_id}')
    depth_anything = DepthAnythingEncoder.from_pretrained('LiheYoung/depth_anything_{:}14'.format(args.dpt_encoder)).eval().to(device)
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    return depth_anything, transform
    
def get_foundation_model(model_name, args=None):
    if model_name == 'midas':
        return make_beit_backbone(output='dense', layer=-1, midas=True)
    elif model_name == 'sam':
        return SAM(arch='vit_l', output='dense', layer=-1)
    elif model_name == 'dinov2':
        return DINO(dino_name='dinov2', model_name='vitl14', output='dense', layer=-1)
    elif model_name == 'mae':
        return MAE(checkpoint='facebook/vit-mae-large', output='dense', layer=-1)
    elif model_name == 'ibot':
        return iBOT(model_type='large_22k', output='dense', layer=-1)
    elif model_name == 'depth_anything':
        model, transform = build_depth_anything(args)
        return model, transform
    else:
        print('No Matching Model')
    

'''
https://github.com/santurini/cosine-annealing-linear-warmup/tree/main
'''
class CosineAnnealingWithWarmup(_LRScheduler):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            first_cycle_steps: int,
            min_lrs: List[float] = None,
            cycle_mult: float = 1.,
            warmup_steps: int = 0,
            gamma: float = 1.,
            last_epoch: int = -1,
            min_lrs_pow: int = None,
                 ):

        '''
        :param optimizer: warped optimizer
        :param first_cycle_steps: number of steps for the first scheduling cycle
        :param min_lrs: same as eta_min, min value to reach for each param_groups learning rate
        :param cycle_mult: cycle steps magnification
        :param warmup_steps: number of linear warmup steps
        :param gamma: decreasing factor of the max learning rate for each cycle
        :param last_epoch: index of the last epoch
        :param min_lrs_pow: power of 10 factor of decrease of max_lrs (ex: min_lrs_pow=2, min_lrs = max_lrs * 10 ** -2
        '''
        assert warmup_steps < first_cycle_steps, "Warmup steps should be smaller than first cycle steps"
        assert min_lrs_pow is None and min_lrs is not None or min_lrs_pow is not None and min_lrs is None, \
            "Only one of min_lrs and min_lrs_pow should be specified"
        
        # inferred from optimizer param_groups
        max_lrs = [g["lr"] for g in optimizer.state_dict()['param_groups']]

        if min_lrs_pow is not None:
            min_lrs = [i * (10 ** -min_lrs_pow) for i in max_lrs]

        if min_lrs is not None:
            assert len(min_lrs)==len(max_lrs),\
                "The length of min_lrs should be the same as max_lrs, but found {} and {}".format(
                    len(min_lrs), len(max_lrs)
                )

        self.first_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle_mult = cycle_mult  # cycle steps magnification
        self.base_max_lrs = max_lrs  # first max learning rate
        self.max_lrs = max_lrs  # max learning rate in the current cycle
        self.min_lrs = min_lrs  # min learning rate
        self.warmup_steps = warmup_steps  # warmup step size
        self.gamma = gamma  # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps  # first cycle step size
        self.cycle = 0  # cycle count
        self.step_in_cycle = last_epoch  # step size of the current cycle

        super().__init__(optimizer, last_epoch)

        assert len(optimizer.param_groups) == len(self.max_lrs),\
            "Expected number of max learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))
        
        assert len(optimizer.param_groups) == len(self.min_lrs),\
            "Expected number of min learning rates provided ({}) to be the same as the number of groups parameters ({})".format(
                len(max_lrs), len(optimizer.param_groups))

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for i, param_groups in enumerate(self.optimizer.param_groups):
            param_groups['lr'] = self.min_lrs[i]
            self.base_lrs.append(self.min_lrs[i])

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(max_lr - base_lr) * self.step_in_cycle / self.warmup_steps + base_lr for (max_lr, base_lr) in
                    zip(self.max_lrs, self.base_lrs)]
        else:
            return [base_lr + (max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle - self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for (max_lr, base_lr) in zip(self.max_lrs, self.base_lrs)]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int(
                    (self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lrs = [base_max_lr * (self.gamma ** self.cycle) for base_max_lr in self.base_max_lrs]
        self.last_epoch = math.floor(epoch)

        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]