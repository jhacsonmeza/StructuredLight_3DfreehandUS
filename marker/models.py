import cv2
import torch
import numpy as np

from marker import utils


class DoubleConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(True)
        )

    def forward(self, x):
        return self.double_conv(x)


class SuperPointNet(torch.nn.Module):
    def __init__(self, Nc):
        super(SuperPointNet, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Shared backbone
        self.conv1 = DoubleConv(1, 64)
        self.conv2 = DoubleConv(64, 64)
        self.conv3 = DoubleConv(64, 128)
        self.conv4 = DoubleConv(128, 128)
        
        # Detector Head.
        self.convD = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        )
        
        # ID classifier Head.
        self.convC = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(256, Nc+1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        # Shared backbone
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        
        # Detector Head.
        det = self.convD(x)
        
        # ID classifier Head.
        cls = self.convC(x)
        
        return det, cls

class EllipSegNet(torch.nn.Module):
    def __init__(self, init_f, num_outputs):
        super(EllipSegNet, self).__init__()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.inc = DoubleConv(1, init_f)
        self.down1 = DoubleConv(init_f, 2*init_f)
        self.down2 = DoubleConv(2*init_f, 4*init_f)
        self.down3 = DoubleConv(4*init_f, 4*init_f)
        self.up1 = DoubleConv(2*4*init_f, 2*init_f, 4*init_f)
        self.up2 = DoubleConv(2*2*init_f, init_f, 2*init_f)
        self.up3 = DoubleConv(2*init_f, init_f)
        self.outc = torch.nn.Conv2d(init_f, num_outputs, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x) #(120,120)
        x2 = self.down1(self.pool(x1)) #(60,60)
        x3 = self.down2(self.pool(x2)) #(30,30)
        x4 = self.down3(self.pool(x3)) #(15,15)
        
        x = torch.cat([self.upsample(x4), x3], 1) #(15*2,15*2), (30,30)
        x = self.up1(x)
        
        x = torch.cat([self.upsample(x), x2], 1) #(30*2,30*2), (60,60)
        x = self.up2(x)
        
        x = torch.cat([self.upsample(x), x1], 1) #(60*2,60*2), (120,120)
        x = self.up3(x)
        
        x = self.outc(x) #(120,120)
        return x


class Detector(torch.nn.Module):
    def __init__(self, detect, ellipseg, imresize, crop_sz):
        super(Detector, self).__init__()
        self.detect = detect
        self.ellipseg = ellipseg

        self.imresize = imresize

        self.crop_sz = crop_sz
        self.mid = (crop_sz-1)//2
    
    def pixelPoints(self, out_det, out_cls):
        scores = utils.labels2scores(out_det)
        scores = utils.simple_nms(scores, 4)
        scores = scores[0]

        # Extract keypoints
        keypoints = torch.nonzero(scores > 0.015, as_tuple=False) #(n,2) with rows,cols
        scores = scores[tuple(keypoints.t())] #(n,1)

        # Keep the 3 keypoints with highest score
        if keypoints.shape[0] > 3:
            scores, indices = torch.topk(scores, 3, dim=0)
            keypoints = keypoints[indices]

        # Class id
        out_cls = out_cls.argmax(1).squeeze(0)
        r,c = (keypoints//8).T
        id_class = out_cls[r,c]

        if torch.any(id_class == 3):
            id_class[id_class == 3] = 6-id_class.sum()

        # Sort keypoints
        keypoints = keypoints[torch.argsort(id_class)].cpu().numpy()

        # from (row,col) to (x,y)
        keypoints = np.fliplr(keypoints)

        # Using Shoelace formula to know orientation of point
        A = keypoints[1,0]*keypoints[2,1]-keypoints[2,0]*keypoints[1,1] - \
            keypoints[0,0]*keypoints[2,1]+keypoints[2,0]*keypoints[0,1] + \
            keypoints[0,0]*keypoints[1,1]-keypoints[1,0]*keypoints[0,1]
        
        if A > 0:
            keypoints = keypoints[[1,0,2]]

        return keypoints
    
    def patches(self, im, keypoints):
        # Estimate top-left point
        tl = np.int32(np.round(keypoints)) - self.mid

        # Create patches
        crops = []
        for i in range(3):
            imcrop = im[tl[i,1]:tl[i,1]+self.crop_sz, tl[i,0]:tl[i,0]+self.crop_sz]
            if imcrop.shape != (self.crop_sz,self.crop_sz):
                inter_wh = np.minimum(im.shape[::-1],tl[i]+self.crop_sz)-np.maximum(0,tl[i])

                cx = self.crop_sz - inter_wh[0]
                cy = self.crop_sz - inter_wh[1]
                if cx != 0: cx += 1
                if cy != 0: cy += 1

                if (keypoints[i,0]+cx+self.mid > im.shape[1]-1) | (keypoints[i,0]+cx-self.mid < 0): cx = -cx
                if (keypoints[i,1]+cy+self.mid > im.shape[0]-1) | (keypoints[i,1]+cy-self.mid < 0): cy = -cy

                # Transform image
                Ht = np.array([[1.,0.,cx],[0.,1.,cy]])
                im2 = cv2.warpAffine(im.copy(),Ht,None,None,cv2.INTER_LINEAR,cv2.BORDER_REPLICATE)

                # Modify tl corner
                c = keypoints[i].copy()
                c[0] += cx
                c[1] += cy
                tl2 = np.int32(np.round(c)) - self.mid

                # Crop
                imcrop = im2[tl2[1]:tl2[1]+self.crop_sz, tl2[0]:tl2[0]+self.crop_sz]
            
            crops.append(imcrop)
        
        crops = np.dstack(crops)

        return crops

    def forward(self, x):
        device = next(self.parameters()).device

        # --------------------------------------------------- Rough 2D point detection
        # Convert image to gray and resize
        imr = cv2.resize(x, self.imresize)

        # Convert image to tensor
        imt = torch.from_numpy(np.float32(imr/255)).unsqueeze(0).unsqueeze(0)
        imt = imt.to(device)

        # Pixel point estimation
        out_det, out_cls = self.detect(imt)
        keypoints = self.pixelPoints(out_det, out_cls)

        # Scale points to full resolution
        keypoints = keypoints/np.array(imr.shape[::-1])*np.array(x.shape[::-1])



        # --------------------------------------------------- Ellipse segmetnation and sub-pixel center estimation
        crops = self.patches(x, keypoints)

        # Convert to tensor
        cropst = torch.from_numpy(np.float32(crops/255))
        cropst = cropst.permute(2,0,1).unsqueeze(1).to(device)

        # Ellipse contour estimation
        out = torch.sigmoid(self.ellipseg(cropst))
        out = out.squeeze(1).detach().cpu().numpy()
        out = np.uint8(255*(out>0.5))
        
        # Ellipse center estimation
        centers = []
        for mask in out:
            contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            idx = 0
            if len(contours) > 1:
                areas = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    areas.append(area)
                idx = np.argmax(areas).item()
            
            rbox = cv2.fitEllipse(contours[idx])
            centers.append([rbox[0][0],rbox[0][1]])

        centers = centers + np.int32(np.round(keypoints)) - self.mid

        return centers
