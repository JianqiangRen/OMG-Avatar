import face_alignment
import cv2
import torch
import torchvision
import numpy as np
from skimage import io, img_as_float

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda:0')

def akd(img1_path, img2_path):
    img1 = io.imread(img1_path)
    img2 = io.imread(img2_path)
    pred1 = fa.get_landmarks(img1)[0]
    pred2 = fa.get_landmarks(img2)[0]

    tensor1 = torch.from_numpy(pred1.astype(np.float32)).cuda()
    tensor2 = torch.from_numpy(pred2.astype(np.float32)).cuda()
    # print(tensor1.shape)
    # print(tensor2.shape)
    akd = torch.nn.functional.l1_loss(tensor1, tensor2)
    return akd.cpu().item()

if __name__ == '__main__':
    img1_path = ''
    img2_path = ''
    akd = akd(img1_path, img2_path)
    print(akd)