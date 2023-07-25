import numpy as np
from tqdm import tqdm
from torch.backends import cudnn
from utils import pyutils
import argparse
import os.path
import cv2
from modeling.deeplab import *
from dataloaders import data_loader, imutils
from torch.utils.data import DataLoader
from torchvision import transforms
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
cudnn.enabled = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # set up parameters
    parser.add_argument("--weights", default=r'', type=str)
    parser.add_argument("--out_dir", default=r'/', type=str)
    parser.add_argument("--data_root", default=r'', type=str)
    parser.add_argument("--batch_size", default=16, type=int)
    args = parser.parse_args()

    # set up dataloader
    matrix = np.array([[3.84124631e-02,1.76116254e-02,1.88580000e+04]
                    ,[4.86044462e-02,1.96986047e-02,1.78265000e+04]
                    ,[4.66588088e-02,2.46997838e-02,1.73660000e+04]
                    ,[1.22656998e-01,5.68736669e-02,1.61636875e+04]
                    ,[1.04468634e-01,5.21267470e-02,1.53575625e+04]
                    ,[7.45429324e-02,4.44962726e-02,1.52129375e+04]])
    val_dataset = data_loader.Dataset_MSS_test(args.data_root, transform=transforms.Compose([
                                        np.asarray,
                                        imutils.Normalize_VAL2(mean=matrix[:,0], std=matrix[:,1], max=matrix[:,2]),
                                        imutils.HWC_to_CHW_VAL,
                                        torch.from_numpy
                                    ]))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=0, pin_memory=True, drop_last=False)

    # load model
    model = DeepLab(num_classes=2,
                    backbone='resnet',
                    output_stride=8,
                    sync_bn=None,
                    freeze_bn=False)

    model.load_state_dict(torch.load(args.weights)['state_dict'],strict=False)
    model.eval()
    model.cuda()

    # output log
    pyutils.Logger(args.out_dir+'print.txt')
    print(vars(args))

    # inference
    tbar = tqdm(val_loader)
    for i, sample in enumerate(tbar):
        name, image = sample[0], sample[1]
        image = image.cuda()
        with torch.no_grad():
            output = model(image)
        output = torch.argmax(output,dim=1).cpu().numpy()
        for i in range(output.shape[0]):
            cv2.imwrite(os.path.join(args.out_dir, name[i]+ '.png'),output[i].astype(np.uint8)*255)
