import numpy as np
import torch
from tqdm import tqdm
from utils import *
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import os
import torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from dataloader import *
from model import *
import torch.distributed as dist

if __name__ == "__main__":
    # 是否使用GPU
    Cuda = True
    num_workers = 4
    distributed = False
    num_classes = 9
    predict_type = "ConfidenceInterval" # ConfidenceInterval or Result
    model_path = r"checkpoints/segmentation/2024-04-04-20-54-26/model_state_dict_loss0.0018_epoch130.pth"

    input_shape = [512, 512]
    output_folder = r"predict"
    data_dir = r"data/predict"

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

    if local_rank == 0:
        print("===============================================================================")

    model = MAE_ViT(image_size=input_shape[0], patch_size=32)      
    model = ViT_Segmentor(model.encoder, out_channels=num_classes, downsample_factor=16)

    if Cuda:
        if distributed:
            model = model.cuda(local_rank)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model = model.cuda()

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model.load_state_dict(torch.load(model_path))  
        
    with open(os.path.join(data_dir, r"list/predict.txt"),"r") as f:
        predict_lines = f.readlines()
    num_predict = len(predict_lines)

    if local_rank == 0:
        print("device:", device, "num_predict:", num_predict)
        print("===============================================================================")

    image_transform = get_transform(input_shape, IsResize=True, IsTotensor=True, IsNormalize=True)

    # 开始模型训练
    if local_rank == 0:
        print("start predicting")

    model.eval()      
    for annotation_line in tqdm(predict_lines):
        annotation_line 
        name_image = annotation_line.split()[0]

        im_data, im_Geotrans, im_proj, cols, rows = read_tif(name_image)
        image = Image.open(name_image)
        name = os.path.basename(name_image)

        if image_transform is not None:
            image = image_transform(image)
            image = image.unsqueeze(0)
            
        else:
            image = torch.from_numpy(np.transpose(np.array(image), [2, 0 ,1]))

        image = image.to(device).float()
        prediction, out, embedding = model(image)  
        prediction = F.softmax(prediction, dim=1)
        if predict_type == "Result":
            prediction = torch.argmax(prediction, dim=1)
        elif predict_type == "ConfidenceInterval":
            prediction = prediction * 100
            prediction = prediction.type(torch.uint8)
            prediction = prediction.squeeze(0)
        if local_rank == 0:
            if not os.path.exists(os.path.join(output_folder, name)):
                write_tif(os.path.join(output_folder, name), prediction.cpu().detach().numpy(), im_Geotrans, im_proj, gdal.GDT_Byte)

    if distributed:
        dist.barrier()

    if local_rank == 0:
        print("finish predicting successfully")
        print("===============================================================================") 

# python predict_segmentor.py
