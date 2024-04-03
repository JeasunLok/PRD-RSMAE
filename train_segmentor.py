import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os
import torch
import torch.nn as nn
import time
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from dataloader import *
from model import *
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.cuda.amp import GradScaler

#-------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, criterion, optimizer, e, epoch, device, num_classes, scaler, fp16):
    loss_show = AverageMeter()
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    loop = tqdm(enumerate(train_loader), total = len(train_loader))
    for batch_idx, (batch_data, batch_label) in loop:
        batch_data = batch_data.to(device).float()
        batch_label = batch_label.to(device).long()

        optimizer.zero_grad()

        if fp16:
            with torch.cuda.amp.autocast():
                batch_prediction = model(batch_data)
                loss = criterion(batch_prediction, batch_label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_prediction = model(batch_data)
            loss = criterion(batch_prediction, batch_label)
            loss.backward()
            optimizer.step()     

        batch_prediction = F.softmax(batch_prediction, dim=1)
        batch_prediction = torch.argmax(batch_prediction, dim=1)
        # calculate the accuracy
        CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
        CM = CM + CM_batch
        n = batch_data.shape[0]

        # update the loss and the accuracy 
        loss_show.update(loss.data, n)
        acc = compute_acc(CM)
        mIoU = compute_mIoU(CM)

        loop.set_description(f'Train Epoch [{e+1}/{epoch}]')
        loop.set_postfix({"train_loss":loss_show.average.item(),
                          "train_accuracy": str(round(acc*100, 2)) + "%",
                          "train_mIoU": str(round(mIoU*100, 2)) + "%"})

    return acc, mIoU, loss_show.average.item()
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# valid model
def valid_epoch(model, val_loader, criterion, e, epoch, device, num_classes):
    loss_show = AverageMeter()
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    loop = tqdm(enumerate(val_loader), total = len(val_loader))
    with torch.no_grad():
        for batch_idx, (batch_data, batch_label) in loop:
            batch_data = batch_data.to(device).float()
            batch_label = batch_label.to(device).long()

            batch_prediction = model(batch_data)
            loss = criterion(batch_prediction, batch_label)

            batch_prediction = F.softmax(batch_prediction, dim=1)
            batch_prediction = torch.argmax(batch_prediction, dim=1)
            # calculate the accuracy
            CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
            CM = CM + CM_batch
            n = batch_data.shape[0]

            # update the loss and the accuracy 
            loss_show.update(loss.data, n)
            acc = compute_acc(CM)
            mIoU = compute_mIoU(CM)

            loop.set_description(f'Val Epoch [{e+1}/{epoch}]')
            loop.set_postfix({"val_loss":loss_show.average.item(),
                            "val_accuracy": str(round(acc*100, 2)) + "%",
                            "val_mIoU": str(round(mIoU*100, 2)) + "%"})

    return CM, acc, mIoU, loss_show.average.item()

#-------------------------------------------------------------------------------
# test model
def test_epoch(model, test_loader, device, num_classes):
    acc = 0
    mIoU = 0
    CM = np.zeros([num_classes, num_classes])
    loop = tqdm(enumerate(test_loader), total = len(test_loader))
    with torch.no_grad():
        for batch_idx, (batch_data, batch_label) in loop:
            batch_data = batch_data.to(device).float()
            batch_label = batch_label.to(device).long()

            batch_prediction = model(batch_data)

            batch_prediction = F.softmax(batch_prediction, dim=1)
            batch_prediction = torch.argmax(batch_prediction, dim=1)

            # calculate the accuracy
            CM_batch = confusion_matrix(batch_prediction.cpu().numpy().flatten(), batch_label.cpu().numpy().flatten(), labels=np.array(range(num_classes)))
            CM = CM + CM_batch

            # update the loss and the accuracy 
            acc = compute_acc(CM)
            mIoU = compute_mIoU(CM)

            loop.set_description(f'Test Epoch')
            loop.set_postfix({"test_accuracy": str(round(acc*100, 2)) + "%",
                              "test_mIoU": str(round(mIoU*100, 2)) + "%"})

    return CM, acc, mIoU
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    num_workers = 4
    distributed = True
    sync_bn = True
    fp16 = True

    num_classes = 9

    model_pretrained = False
    model_path = r""
    encoder_pretrained = True
    encoder_path = r""
    

    input_shape = [512, 512]
    epoch = 100
    save_period = 5
    batch_size = 48

    # 学习率
    lr = 5e-4
    min_lr = lr*0.01

    # 优化器
    momentum = 0.9 
    weight_decay = 0
    
    data_dir = r"data/segmentation"
    logs_dir = r"logs"
    checkpoints_dir = r"checkpoints"
    time_now = time.localtime()
    logs_folder = os.path.join(logs_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    checkpoints_folder = os.path.join(checkpoints_dir, time.strftime("%Y-%m-%d-%H-%M-%S", time_now))
    os.makedirs(logs_folder)
    os.makedirs(checkpoints_folder)

    dice_loss = False
    focal_loss = False

    print("===============================================================================")
    # 设置用到的显卡
    ngpus_per_node  = torch.cuda.device_count()
    if Cuda:
        if distributed:
            local_rank = int(os.environ["LOCAL_RANK"])
            init_ddp(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            local_rank = 0
    else:
        device = torch.device("cpu")
        local_rank = 0

    model = MAE_Encoder(image_size=input_shape[0], patch_size=32)
    if encoder_pretrained:
        model.load_state_dict(torch.load(encoder_path)) 
    model = ViT_Segmentor(model.encoder, out_channels=num_classes, downsample_factor=16)

    # 混精度
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model_train)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    if model_pretrained and model_path != '':
        print('Load weights {}.'.format(model_path))
        model_train.load_state_dict(torch.load(model_path))  
        
    with open(os.path.join(data_dir, r"list/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(data_dir, r"list/val.txt"),"r") as f:
        val_lines = f.readlines()
    with open(os.path.join(data_dir, r"list/test.txt"),"r") as f:
        test_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)
    num_test = len(test_lines)

    torch.manual_seed(3407)
    np.random.seed(3407)

    print("device:", device, "num_train:", num_train, "num_val:", num_val, "num_test:", num_test)
    print("===============================================================================")

    # 优化器
    optimizer = torch.optim.Adam(model_train.parameters(), lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch//10, gamma=0.9) 
    criterion = nn.CrossEntropyLoss(ignore_index=0).cuda()

    image_transform = get_transform(input_shape, IsResize=True, IsTotensor=True, IsNormalize=True)
    label_transform = get_transform(input_shape, IsResize=True, IsTotensor=True, IsNormalize=False)
    train_dataset = SegDataset(train_lines, input_shape, num_classes, image_transform=image_transform, label_transform=label_transform)
    val_dataset = SegDataset(val_lines, input_shape, num_classes, image_transform=image_transform, label_transform=label_transform)
    test_dataset = SegDataset(test_lines, input_shape, num_classes, image_transform=image_transform, label_transform=label_transform)

    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        test_sampler     = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False,)
        batch_size      = batch_size // ngpus_per_node
        shuffle         = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle = True

    train_loader = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=val_sampler)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True, drop_last=True, sampler=test_sampler)

    # 开始模型训练
    print("start training")
    epoch_result = np.zeros([4, epoch])
    for e in range(epoch):
        model_train.train()
        train_acc, train_mIoU, train_loss = train_epoch(model_train, train_loader, criterion, optimizer, e, epoch, device, num_classes, scaler, fp16)
        scheduler.step()
        print("Epoch: {:03d} | train_loss: {:.4f} | train_acc: {:.2f}% | train_mIoU: {:.2f}%".format(e+1, train_loss, train_acc*100, train_mIoU*100))
        epoch_result[0][e], epoch_result[1][e], epoch_result[2][e], epoch_result[3][e]= e+1, train_loss, train_acc*100, train_mIoU*100

        if ((e+1) % save_period == 0) | (e == epoch - 1):
            print("===============================================================================")
            print("start validating")
            model_train.eval()      
            val_CM, val_acc, val_mIoU, val_loss = valid_epoch(model_train, val_loader, criterion, e, epoch, device, num_classes)
            val_weighted_recall, val_weighted_precision, val_weighted_f1 = compute_metrics(val_CM)
            if (e != epoch -1):
                print("Epoch: {:03d}  =>  Accuracy: {:.2f}% | MIoU: {:.2f}% | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(e+1, val_acc*100, val_mIoU*100, val_weighted_recall, val_weighted_precision, val_weighted_f1))
            torch.save(model_train, os.path.join(checkpoints_folder, "model_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth"))
            torch.save(model_train.state_dict(), os.path.join(checkpoints_folder, "model_state_dict_loss" + str(round(val_loss, 4)) + "_epoch" + str(e+1) + ".pth"))
            print("===============================================================================")
    
    if distributed:
        train_sampler.set_epoch(epoch)

    if distributed:
        dist.barrier()

    draw_result_visualization(logs_folder, epoch_result)
    print("save train logs successfully")
    print("===============================================================================")

    print("start testing")
    model_train.eval()
    test_CM, test_acc, test_mIoU, val_loss = test_epoch(model_train, val_loader, device)
    test_weighted_recall, test_weighted_precision, test_weighted_f1 = compute_metrics(test_CM)
    print("Test Result  =>  Accuracy: {:.2f}%| mIoU: {:.2f} | W-Recall: {:.4f} | W-Precision: {:.4f} | W-F1: {:.4f}".format(test_acc*100, test_mIoU*100, test_weighted_recall, test_weighted_precision, test_weighted_f1))
    store_result(logs_folder, test_acc, test_mIoU, test_weighted_recall, test_weighted_precision, test_weighted_f1, test_CM, epoch, batch_size, lr, weight_decay)
    print("save test result successfully")
    print("===============================================================================") 


