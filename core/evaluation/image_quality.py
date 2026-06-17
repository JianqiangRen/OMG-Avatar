import lpips
import numpy as np
import torch
import math
import cv2
import torchvision
import torchmetrics
import sys
sys.path.append('./')
import core.evaluation.pytorch_ssim as pytorch_ssim

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# loss_fn_alex = lpips.LPIPS(net='alex').to(device) # best forward scores
loss_fn_squeeze = lpips.LPIPS(net='squeeze').to(device)
# loss_fn_vgg = lpips.LPIPS(net='vgg').to(device) # closer to "traditional" perceptual loss, when used for optimization
criterion_ssim = pytorch_ssim.SSIM().to(device)

# def cal_lpips_alex(img1, img2):
#     '''calculate LIPIS

#      img1, img2: [0, 255] ,RGB
#      '''
#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)

#     tensor1 = torch.from_numpy(img1).to(device)
#     tensor2 = torch.from_numpy(img2).to(device)

#     # image should be RGB, IMPORTANT: normalized to [-1,1]
#     tensor1 = (tensor1 /255.0 -0.5) *2
#     tensor2 = (tensor2 /255.0 -0.5) *2
    
#     # tensor1 = tensor1 /255.0
#     # tensor2 = tensor2 /255.0

#     tensor1 = tensor1.permute((2, 0, 1)).unsqueeze(0)
#     tensor2 = tensor2.permute((2, 0, 1)).unsqueeze(0)

#     with torch.no_grad():
#         lpips_value = loss_fn_alex(tensor1, tensor2)

#     return lpips_value.cpu().item()

def cal_lpips_squeeze(img1, img2):
    '''calculate LIPIS

     img1, img2: [0, 255] ,RGB
     '''
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    tensor1 = torch.from_numpy(img1).to(device)
    tensor2 = torch.from_numpy(img2).to(device)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    tensor1 = (tensor1 /255.0 -0.5) *2
    tensor2 = (tensor2 /255.0 -0.5) *2
    
    # tensor1 = tensor1 /255.0
    # tensor2 = tensor2 /255.0

    tensor1 = tensor1.permute((2, 0, 1)).unsqueeze(0)
    tensor2 = tensor2.permute((2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        lpips_value = loss_fn_squeeze(tensor1, tensor2)

    return lpips_value.cpu().item()

# def cal_lpips_vgg(img1, img2):
#     '''calculate LIPIS

#      img1, img2: [0, 255] ,RGB
#      '''
#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)

#     tensor1 = torch.from_numpy(img1).to(device)
#     tensor2 = torch.from_numpy(img2).to(device)

#     # image should be RGB, IMPORTANT: normalized to [0,1]
#     tensor1 = tensor1/255.0
#     tensor2 = tensor2/255.0

#     tensor1 = tensor1.permute((2, 0, 1)).unsqueeze(0)
#     tensor2 = tensor2.permute((2, 0, 1)).unsqueeze(0)

#     with torch.no_grad():
#         lpips_value = loss_fn_vgg(tensor1, tensor2)

#     return lpips_value.cpu().item()

def cal_ssim(img1, img2):
    '''calculate LIPIS

     img1, img2: [0, 255] ,RGB
     '''
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    tensor1 = torch.from_numpy(img1).to(device)
    tensor2 = torch.from_numpy(img2).to(device)

    # image should be RGB, IMPORTANT: normalized to [-1,1]
    tensor1 = (tensor1 /255.0 -0.5) *2
    tensor2 = (tensor2 /255.0 -0.5) *2

    tensor1 = tensor1.permute((2, 0, 1)).unsqueeze(0)
    tensor2 = tensor2.permute((2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        ssim_value = criterion_ssim((tensor1+1.0)/2,(tensor2+1.0)/2)

    return ssim_value.cpu().item()

def cal_psnr(img1, img2):
    '''calculate PSNR

    img1, img2: [0, 255]
    '''
    _img1 = img1.astype(np.float32)
    _img2 = img2.astype(np.float32)
    mse = np.mean((_img1 - _img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def psnr_ssim_lpips(img1_path, img2_path):
    img1 = cv2.imread(img1_path)[...,::-1]
    img2 = cv2.imread(img2_path)[...,::-1]
    psnr = cal_psnr(img1, img2)
    ssim = cal_ssim(img1, img2)
    # lpips = cal_lpips_alex(img1, img2)
    # lpips = cal_lpips_squeeze(img1, img2)
    # lpips = cal_lpips_vgg(img1, img2)
    # lpips, ssim = cal_lpips_and_ssim(img1, img2)
    tensor1 = lpips.im2tensor(lpips.load_image(img1_path)).cuda()
    tensor2 = lpips.im2tensor(lpips.load_image(img2_path)).cuda()
    with torch.no_grad():
        # lpips_value = loss_fn_alex(tensor1, tensor2)
        lpips_value = loss_fn_squeeze(tensor1, tensor2)
    lpips_v = lpips_value.cpu().item()
    
    return psnr, ssim, lpips_v

if __name__ == '__main__':
    img1_path = './data/VFHQ_test_50clips_sample50/Clip+_HebIzK_LP4+P2+C1+F16589-16715/00000000.png'
    img2_path = './data/VFHQ_test_50clips_sample50/Clip+_HebIzK_LP4+P2+C1+F16589-16715/00000002.png'
    psnr, ssim, lpips = psnr_ssim_lpips(img1_path, img2_path)
    print(lpips)
    print(ssim)
    print(psnr)