"""
• Geometric Transformations. Geometric transformations involve altering the spatial prop-
erties of data, such as applying random cropping, rotations, scaling, and flipping. These
transformations may help simulate different viewpoints or object orientations that the
robotic systems might encounter, enhancing the policy’s ability to adapt to spatial varia-
tions during task execution.

• Color Jittering. Color jittering applies adjustments to the brightness, contrast, hue, and
saturation of images. These operations mimic changes in lighting conditions or camera
imperfections, ensuring that the policy can robustly handle various visual environments
that differ in illumination or color properties.

• Adversary Perturbation Adversary perturbation involves adding noise or distortions to
the input data, such as Gaussian noise or pixel-level perturbations. This technique aims
to increase robustness by preparing the policy to perform well under challenging circum-
stances, including sensor noise or environmental variations, that could corrupt the visual or
sensory input.

• Background Augmentation Background augmentation focuses on modifying or replacing
the backgrounds of the input data, either synthetically or by compositing with new random
backgrounds. This technique is often employed to make the policy invariant to irrelevant
environmental features, ensuring robust object detection or manipulation regardless of sur-
rounding visual clutter.

"""
import torch
import numpy as np
import torchvision.transforms.functional as TF
import random
from PIL import Image
from pathlib import Path
import os

class GeometricTransformationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_aug=0.5, max_rotate=10, max_translate=0.1, min_scale=0.8, max_scale=1.2,fill=0):
        super().__init__()
        self.dataset = dataset
        self.p_aug = p_aug
        self.max_rotate = max_rotate
        self.max_translate = max_translate
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.fill = fill
    
    def random_affine_and_pad(
        self,
        images,
        max_rotate=10,      # 度
        max_translate=0.1,  # 相对宽高比例
        min_scale=0.8, max_scale=1.2,
        fill=0
        ):
        """
        images: torch.Tensor, shape [K, C, H, W], dtype uint8
        return: torch.Tensor, shape [K, C, H, W], dtype uint8
        """
        K, C, H, W = images.shape

        # 1. 转为 float32 并归一化
        imgs = images.float() / 255.0

        # 2. 随机采样一套参数（对所有 K 张图共享）
        angle   = random.uniform(-max_rotate, max_rotate)
        tx      = random.uniform(-max_translate * W, max_translate * W)
        ty      = random.uniform(-max_translate * H, max_translate * H)
        scale   = random.uniform(min_scale, max_scale)

        augmented = []
        for img in imgs:  # img: [C, H, W]
            # 3. 先做仿射（旋转+平移+缩放）
            img = TF.affine(img,
                            angle=angle,
                            translate=[tx, ty],
                            scale=scale,
                            shear=0,
                            interpolation=TF.InterpolationMode.BILINEAR,
                            fill=fill)        # 此时尺寸仍为 H×W

            # 4. 如果实际变换后内容小于原图，中心裁剪保证一定是 H×W
            #    （如果缩放>1.0，内容会超出，这里再裁剪回来）
            _, h_, w_ = img.shape
            if h_ != H or w_ != W:
                # 计算左上角，使中心对齐
                top  = (h_ - H) // 2
                left = (w_ - W) // 2
                img  = TF.crop(img, top, left, H, W)

            augmented.append(img)

        # 5. 合并并转回 uint8
        out = torch.stack(augmented, dim=0)          # [K, C, H, W]
        out = (out * 255).clamp(0, 255).to(torch.uint8)
        return out
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        # Perform Transformation
        image = sample['image'] # k*c*h*w
        if random.rand()<self.p_aug: sample['image'] = self.random_affine_and_pad(image, self.max_rotate, self.max_translate, self.min_scale, self.max_scale, self.fill)
        return sample
        
class ColorJitteringDataset(torch.utils.data.Dataset):
    def __init__(self, 
        dataset,  
        p_aug=0.5, 
        brightness=0.2,   # 亮度变化强度，0 表示不变
        contrast=0.2,     # 对比度变化强度
        saturation=0.2,   # 饱和度变化强度
        hue=0.05,         # 色相变化强度，0~0.5
        highlight=0.3   # 高光强度，0 表示无高光
    ):
        super().__init__()
        self.dataset = dataset
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.highlight = highlight

    def random_photometric(
        self,
        images,
        brightness=0.2,   # 亮度变化强度，0 表示不变
        contrast=0.2,     # 对比度变化强度
        saturation=0.2,   # 饱和度变化强度
        hue=0.05,         # 色相变化强度，0~0.5
        highlight=0.3   # 高光强度，0 表示无高光
        ):   
        """
        images: torch.Tensor, shape [K, C, H, W], dtype uint8
        return : torch.Tensor, shape [K, C, H, W], dtype uint8
        """
        K, C, H, W = images.shape
        device = images.device           # CPU 或 CUDA

        # 1. uint8 -> float32 [0,1]
        x = images.float() / 255.0

        # 2. 生成整批共享的随机参数
        b = 1.0 + random.uniform(-brightness, brightness)
        c = 1.0 + random.uniform(-contrast,    contrast)
        s = 1.0 + random.uniform(-saturation,  saturation)
        h = random.uniform(-hue, hue)

        # 3. 颜色抖动（亮度 / 对比度 / 饱和度 / 色相）
        #    先拼成 [K, C, H, W] -> 逐张应用 F.adjust_* 即可
        x = TF.adjust_brightness(x, b)
        x = TF.adjust_contrast(x, c)
        x = TF.adjust_saturation(x, s)
        x = TF.adjust_hue(x, h)

        # 4. 高光（模拟过曝）：把像素值整体抬高后 clamp，再可选 gamma 压缩
        if highlight > 0:
            boost = random.uniform(0, highlight)
            x = x + boost            # 过曝
            x = torch.clamp(x, 0, 1)

            # 可再加一条 gamma 曲线让亮部更柔和（可选）
            gamma = random.uniform(0.7, 1.3)
            x = x.pow(gamma)

        # 5. float32 -> uint8
        out = (x * 255).clamp(0, 255).to(torch.uint8)
        return out
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        # Perform Transformation
        image = sample['image'] # k*c*h*w
        if random.rand()<self.p_aug: sample['image'] = self.random_photometric(image, self.brightness, self.contrast, self.saturation, self.hue, self.highlight)
        return sample
         
class AdversaryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_aug=0.5, ps=[0.25, 0.25, 0.25, 0.25], eps=6/255, alpha=0.15, seed=None):
        super().__init__()
        self.dataset = dataset
        self.p_aug = p_aug
        self.ps = np.array(ps)
        self.ps /= self.ps.sum()
        self.eps = eps
        self.alpha = alpha
        self.seed = seed
    
    def adversarial_aug(
            self,
            images,
            eps=6/255,
            alpha=0.15,
            seed=None
        ):
        """
        images: torch.Tensor [K,C,H,W] uint8
        返回: 同形状 uint8
        """
        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        device = images.device
        K, C, H, W = images.shape
        x = images.float() / 255.0          # [0,1]
        p_noise, p_stripe, p_checker, p_patch = self.ps
        # 决定每张图的扰动类型
        types = random.choices(
            ['noise', 'stripe', 'checker', 'patch'],
            weights=[p_noise, p_stripe, p_checker, p_patch],
            k=K
        )

        out = []
        for img, t in zip(x, types):        # img [C,H,W]
            if t == 'noise':
                delta = (torch.rand_like(img) * 2 - 1) * eps
                img = torch.clamp(img + delta, 0, 1)

            elif t == 'stripe':
                theta = random.random() * 2 * math.pi
                xv, yv = torch.meshgrid(
                    torch.linspace(-1, 1, W, device=device),
                    torch.linspace(-1, 1, H, device=device),
                    indexing='xy')
                stripe = torch.sin(xv * math.cos(theta) + yv * math.sin(theta) * 30)
                delta = stripe * alpha * (random.choice([-1, 1]))
                img = torch.clamp(img + delta.unsqueeze(0), 0, 1)

            elif t == 'checker':
                fw = random.randint(4, 16)
                fh = random.randint(4, 16)
                pattern = ((torch.arange(H)[:, None] // fh) +
                        (torch.arange(W)[None, :] // fw)) % 2
                delta = pattern.to(device).float() * alpha * random.choice([-1, 1])
                img = torch.clamp(img + delta.unsqueeze(0), 0, 1)

            elif t == 'patch':
                ph = random.randint(H//8, H//2)
                pw = random.randint(W//8, W//2)
                top = random.randint(0, H - ph)
                left = random.randint(0, W - pw)
                delta = torch.zeros(C, H, W, device=device)
                delta[:, top:top+ph, left:left+pw] = random.choice([-alpha, alpha])
                img = torch.clamp(img + delta, 0, 1)

            out.append(img)

        out = torch.stack(out, 0)
        return (out * 255).clamp(0, 255).to(torch.uint8)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        # Perform Transformation
        image = sample['image'] # k*c*h*w
        if random.rand()<self.p_aug: sample['image'] = self.adversarial_aug(image, self.eps, self.alpha, self.seed)
        return sample
    
class BGAugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, p_aug=0.5, bg_dir=None):
        super().__init__()
        self.dataset = dataset
        self.p_aug = p_aug
        self.bg_dir = bg_dir
        self.bg_list = self.list_bg_files(self.bg_dir)

    def list_bg_files(self, root):
        root = Path(root)
        IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        return [str(p) for p in root.rglob('*') if p.suffix.lower() in IMG_EXT]
    
    def replace_background_batch(self, images, masks, bg_root):
        """
        images : torch.Tensor [K, C, H, W] uint8
        masks  : torch.Tensor [K, H, W]   bool   True=前景
        bg_root: str 背景图文件夹
        return : torch.Tensor [K, C, H, W] uint8
        """
        assert images.shape[0] == masks.shape[0]
        K, C, H, W = images.shape
        device = images.device
        out = []
        for k in range(K):
            # 前景 & mask
            fg = images[k]               # [C,H,W]
            mask = masks[k]              # [H,W] bool
            mask3 = mask.unsqueeze(0).expand(C, -1, -1)   # [C,H,W]
            # 2. 随机选一张背景
            bg_path = random.choice(self.bg_list)
            bg = Image.open(bg_path).convert('RGB')
            # 3. 等比缩放 + 中心裁剪到 [H,W]
            bg = TF.to_tensor(bg)        # [3,?,?] 0~1
            _, h_bg, w_bg = bg.shape
            scale = max(H / h_bg, W / w_bg)
            new_h, new_w = int(h_bg * scale), int(w_bg * scale)
            bg = TF.resize(bg, [new_h, new_w], antialias=True)
            # 中心裁剪
            top  = (new_h - H) // 2
            left = (new_w - W) // 2
            bg = TF.crop(bg, top, left, H, W)
            # 4. 组合
            bg = (bg * 255).to(torch.uint8).to(device)   # [3,H,W]
            merged = torch.where(mask3, fg, bg)
            out.append(merged)
        return torch.stack(out, 0)   # [K,C,H,W] uint8
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        # Perform Transformation
        assert 'mask' in sample, "Foreground Mask Not Found"
        if random.rand()<self.p_aug:
            image = sample['image'] # k*c*h*w
            mask = sample['mask'] # k*h*w
        return sample
    
