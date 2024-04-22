import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from utils.data_loading import BasicDataset
from utils.dataset_collate import my_collate
# from utils.dice_score import dice_loss
from utils.dice import SoftDiceLoss
from evaluate import evaluate
import time

from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm

from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from albumentations import RandomRotate90,Resize,HorizontalFlip,Flip, VerticalFlip
import copy
sys.path.append(r"./models/unet/")
from models.unet.unet import UNet
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
from topologylayer.nn import AlphaLayer, BarcodePolyFeature
# from models.SGCN.SGCNNet import SGCN_res50
from topolayer_new import TopLoss
from topolayer_new_circle import TopCircleLoss
from topolayer_new_loop import TopLoopLoss

# 不变的路径放到最上面
train_dir_img = Path('/home/sunsong/0-dataset/CAMUS/sep/train/imgs/')
train_dir_mask = Path('/home/sunsong/0-dataset/CAMUS/sep/train/masks/')

val_dir_img = Path('/home/sunsong/0-dataset/CAMUS/sep/val/imgs/')
val_dir_mask = Path('/home/sunsong/0-dataset/CAMUS/sep/val/masks/')

dir_checkpoint = Path('./result/pth/')
writer = SummaryWriter(
    log_dir="./result/runs/",
)



def train_net(net,
              net_topo,
              device,
              n_channels=1,
              n_classes=1,
              epochs: int = 5,
              batch_size: int = 2,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,
              img_scale = (224, 224),
              amp: bool = False):
    # if isinstance(net, torch.nn.DataParallel):
    #     n_channels = net.module.n_channels
    # else:
    #     n_channels = net.n_channels
    # assert val_percent > 0 and val_percent < 100  # 必须保证验证集有数据 否则会报错！！

    ## 0. 创建albumentations的数据增强
    # config = vars(parse_args())
    train_transform = Compose([
        # RandomRotate90(),
        # Flip(),
        Resize(img_scale[0], img_scale[1]),#(h,w)
        # transforms.Normalize(),#这个就不要用了，因为是为三通道设计的，会出现广播错误，一定要用的话就把image repeat成三通道
    ])

    val_transform = Compose([
        Resize(img_scale[0], img_scale[1]),#(h,w)
        # transforms.Normalize(),#这个就不要用了，因为是为三通道设计的，会出现广播错误，一定要用的话就把image repeat成三通道
    ])

    # 1. Create dataset
    train_set = BasicDataset(train_dir_img, train_dir_mask, img_scale,transform=train_transform)
    val_set = BasicDataset(val_dir_img, val_dir_mask, img_scale,transform=val_transform)
    n_train = len(train_set)#这个是没有算batchsize的
    n_val = len(val_set)


    # 2-1. Split into train / validation partitions
    # n_val = int(len(dataset) * val_percent)
    # n_train = len(dataset) - n_val
    # train_set, val_set = random_split(
    #     dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 2-2. 分别读取训练集和验证集


    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=10,
                       pin_memory=False, collate_fn=my_collate)
    train_loader = DataLoader(train_set, shuffle=True,drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False,
                            drop_last=False, **loader_args)

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(
        net.parameters(), lr=learning_rate, weight_decay=1e-8)
    optimizer_topo = optim.Adam(
        net_topo.parameters(), lr=learning_rate, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=10)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    
    ################################################
    ########### 定义loss#############################
    if n_classes == 1:
        criterion = nn.BCELoss(reduction="none")
    else:
        criterion = nn.CrossEntropyLoss()
    
    # criterion = SoftDiceLoss()
    tloss = TopLoss(None)
    # t_circle_loss = TopCircleLoss(None)#size不用设置
    t_circle_loss = TopLoopLoss(None)#size不用设置
    ################################################
    
    layer = AlphaLayer(maxdim=1)
    # f1 = BarcodePolyFeature(0,2,0)
    f2 = BarcodePolyFeature(1,2,0)#正
    
    global_step = 0
    
    min_val_loss = np.inf
    max_val_score = 0
    
    # torch_kernel = torch.from_numpy(np.expand_dims(np.expand_dims(np_kernel,axis=0), axis=0)).to(dtype=torch.float32)
    conv_op = torch.nn.functional.conv2d
    
        
    # 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                # if batch is None:  # 去掉空数据
                #     print("1")
                #     continue
                images = batch['image']
                mask_true = batch['mask']

                assert images.shape[1] == n_channels, \
                    f'Network has been defined with {n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                mask_true = mask_true.to(device=device, dtype=torch.long)

                ########################################################
                with torch.cuda.amp.autocast(enabled=amp):
                    mask_true_onehot = torch.unsqueeze(mask_true, 1)
                    mask_true_onehot = mask_true_onehot.to(dtype=torch.float32).contiguous()
                    
                    topo_loss = torch.tensor(0.).to(dtype=torch.float, device=mask_true.device)
                    new_start = time.time()
                    if n_classes == 1:
                        ##  这里太费时间了  可以只选择中间的像素 周围的也没太大用

                        only_pixel_epoch_num = 1
                        alpha = 1
                        mask_weight = torch.zeros_like(images, device=images.device)#累加每次概率变化绝对值 大的说明更得关注
                        net_topo.load_state_dict(net.state_dict())
                        for _ in range(1):
                            if epoch > only_pixel_epoch_num:
                                topo_loss = torch.tensor(0).to(dtype=torch.float,device=mask_true.device)
                                
                                mask_pred = net_topo(images)
                                mask_pred_activate = F.sigmoid(mask_pred)
                                
                                last_mask_pred_activate = mask_pred_activate

                                mask_pred_activate_resize = nn.functional.interpolate(mask_pred_activate, (64, 64))
                                mask_true_onehot_resize = nn.functional.interpolate(mask_true_onehot, (64, 64))
                                # try:
                                topo_loss_list = []
                                for idx in range(len(mask_pred_activate)):
                                    topo_loss_list.append(tloss(mask_pred_activate_resize[idx, 0].contiguous(),mask_true_onehot_resize[idx, 0].contiguous()))
                                # except:
                                #     break   
                                
                                topo_loss = torch.stack(topo_loss_list, dim=0).sum(dim=0)
                                topo_loss = topo_loss*alpha

                            
                                grad_scaler.scale(topo_loss).backward()
                                grad_scaler.step(optimizer_topo)
                                grad_scaler.update()
                                optimizer_topo.zero_grad()
                                
                                ##  这里实际上只有再前向之后算一遍才会更改坐标
                                topo_loss = torch.tensor(0.).to(dtype=torch.float,device=mask_true.device)
                                
                                # del mask_pred
                                with torch.no_grad():##这里如果不去掉梯度就会不停累加
                                    mask_pred = net_topo(images)
                                    mask_pred_activate = F.sigmoid(mask_pred)
                                    
                                    ## 权重累加与归一化
                                    mask_weight += torch.abs(mask_pred_activate - last_mask_pred_activate)

                        mask_weight = mask_weight * mask_true_onehot            
                        if mask_weight.max() > mask_weight.min():
                            for idx in range(len(mask_weight)):
                                mask_weight[idx] = (mask_weight[idx] - mask_weight[idx].min())/(mask_weight[idx].max() - mask_weight[idx].min())
                                
                                #del mask_pred, last_mask_pred_sigmoid, mask_pred_sigmoid
                                # torch.cuda.empty_cache()
                                
                                
                                
                        ## pixel loss  
                        mask_pred = net(images)      
                        mask_pred_activate = F.sigmoid(mask_pred)
                        
                        mask_true_onehot_resize = nn.functional.interpolate(mask_true_onehot, (64, 64))
                        mask_pred_activate_resize = nn.functional.interpolate(mask_pred_activate, (64, 64))                        
                        
                        mask_weight_cut = (mask_weight>=1e-4)*mask_weight
                        
                        final_mask_weight = mask_weight_cut.detach() + 1
                        
                        all_pixel_loss = criterion(mask_pred_activate, mask_true_onehot)
                        
                        all_pixel_loss_weight = all_pixel_loss * final_mask_weight
                        
                        pixel_loss = all_pixel_loss_weight.mean()
                        t_c_loss_list = []
                        t_c_loss = torch.tensor(0., device=mask_true_onehot.device)
                        if epoch > only_pixel_epoch_num:
                            # try:
                            for idx in range(len(mask_pred_activate_resize)):
                                t_c_loss_list.append(t_circle_loss(predict_map=mask_pred_activate_resize[idx, 0], true_map=mask_true_onehot_resize[idx, 0].to(dtype=torch.int)))
                            t_c_loss = torch.stack(t_c_loss_list, dim=0).sum(dim=0)
                            # except:
                            #     print("t_c_loss 算错了")
                        
                        ## loss不能有inf  否则会产生/opt/conda/conda-bld/pytorch_1670525552843/work/aten/src/ATen/native/cuda/Loss.cu:92: operator(): block: [134,0,0], thread: [127,0,0] Assertion `input_val >= zero && input_val <= one` failed.
                        # 
                        Belta = 0.9
                        total_loss = t_c_loss*(1-Belta) +  pixel_loss*Belta
                        
                        grad_scaler.scale(total_loss).backward()
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        optimizer.zero_grad()
                    
                    #print(save_topo_loss.data/(len(mask_pred)), pixel_loss.data)
                    new_end = time.time()
                    
                    print("拓扑耗时：   {}  秒".format(new_end-new_start))
                    ########################################################


                    pbar.update(images.shape[0])
                    global_step += 1
                    epoch_loss += total_loss.item()

                    # pbar.set_postfix(**{'loss (batch)': loss.item()})
                    pbar.set_postfix(**{'loss (batch)': '{0:1.4f}'.format(total_loss.item())})
                    
                    # Evaluation round
                    division_step = (n_train // batch_size)
                    if division_step > 0:
                        tmp_count = (division_step//5)
                        if global_step % tmp_count == 0:
                            # 增加tensorboard 训练loss 不需要记录训练精度
                            print("训练loss {0:1.4f}".format(total_loss.item()))
                            writer.add_scalar('train/loss',t_c_loss.item()/len(mask_pred),global_step//tmp_count)
                            writer.add_scalar('train/topo_loss',topo_loss.data/(len(mask_pred)),global_step//tmp_count)
                            writer.add_scalar('train/pixel_loss',pixel_loss.data,global_step//tmp_count)
                    
                    # break
        
        # 验证函数 注意必须有验证集          
        val_score, val_loss = evaluate(net=net,n_classes=n_classes, dataloader=val_loader, device=device,criterion=criterion, epoch=epoch, only_pixel_epoch_num=only_pixel_epoch_num)
        scheduler.step(val_loss)


        # 增加tensorboard 验证精度 acc和loss就是泛指精度和损失，通常不需要改变
        writer.add_scalar('val/acc',
                            val_score, epoch)
        writer.add_scalar('val/loss',
                            val_loss.item(), epoch)
        writer.add_scalar('val/lr',
                            scheduler._last_lr[0], epoch)

        # if save_checkpoint:
        ## 保存验证集最优模型
        if val_loss <= min_val_loss:
        # if val_score >= max_val_score:
            min_val_loss = val_loss
            # max_val_score = val_score
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            
            ## 删掉之前保存的模型
            curr_file_list = os.listdir(dir_checkpoint)
            for curr_file in curr_file_list:
                os.remove(os.path.join(dir_checkpoint, curr_file))
            
            ## 保存最优模型
            torch.save(net.state_dict(), str(dir_checkpoint /
                       'checkpoint_epoch_{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(
        description='Train the UNet on images and target mask')
    parser.add_argument('--epochs', '-e', metavar='E',
                        type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size',
                        metavar='B', type=int, default=20, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-3,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str,
                        default=False, help='Load model from a .pth file')
    # parser.add_argument('--scale', '-s', type=float, default=0.4,
                        # help='Downscaling factor of the images')
    # parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        # help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true',
                        default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true',
                        default=False, help='Use bilinear upsampling')
    # parser.add_argument('--classes', '-c', type=int,
    #                     default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args.amp = False##如果loss一开始就不变 就改成False  否则就是True
    device_ids = [3]
    scale = (224, 224)#(H,W)
    n_channels = 1
    n_classes = 1  #原来是因为loss计算有问题 导致错误 一直找不出来
    
    device = torch.device("cuda:{}".format(
        device_ids[0]) if torch.cuda.is_available() else "cpu")
    args.load = r"./result/pth/checkpoint_epoch_45.pth"
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=args.bilinear)#这个模型用amp之后loss也是正常变化的
    # net = SGCN_res50(n_classes=n_classes)#只有这个模型用amp会导致loss一直不变
    net_topo = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=args.bilinear)
    logging.info(f'Network:\n'
                 f'\t{n_channels} input channels\n'
                 f'\t{n_classes} output channels (classes)\n')
                #  f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net_topo = torch.nn.DataParallel(net_topo, device_ids=device_ids)
    
    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    net_topo.to(device=device)

    try:
        train_net(net=net,
                  net_topo = net_topo,
                n_channels=n_channels,
                n_classes=n_classes,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=scale,
                #   val_percent=args.val / 100,
                  amp=args.amp)
        torch.save(net.state_dict(), './result/pth/last_{}.pth'.format(args.epochs))
    except KeyboardInterrupt:
        torch.save(net.state_dict(), './result/pth/INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise
