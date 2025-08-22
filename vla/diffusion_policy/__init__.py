from .diffusion_policy import DiffusionPolicyConfig, DiffusionPolicyModel
from .utils import data_collator
import torch
from .trainer import Trainer
import torchvision.transforms as transforms

def load_model(args):
    if not args.is_pretrained:
        image_sizes = []
        for cam_name in args.camera_names:
            if 'primary' in cam_name:
                image_sizes.append(args.image_size_primary)
            else:
                image_sizes.append(args.image_size_wrist)
        config = DiffusionPolicyConfig(
            camera_names=args.camera_names, 
            image_sizes=image_sizes,
            observation_horizon=1, 
            action_dim=args.action_dim, 
            state_dim = args.state_dim, 
            prediction_horizon = args.chunk_size
        ) 
        model = DiffusionPolicyModel(config=config)
    else:
        model = DiffusionPolicyModel.from_pretrained(args.model_name_or_path, trust_remote_code=True)
        model_args = getattr(args, 'model_args', {})
        # ema
        if model_args.get('using_ema', False):
            model.ema.copy_to(model.parameters()) # using ema for testing
        model.config.num_inference_timesteps = model_args.get('num_inference_steps', 10)
        model.to('cuda')
    # model.to(dtype=torch.float32, device=args.device)
    return {'model': model}

def get_data_collator(args, model_components):
    return data_collator

class DataTransform:
    def __init__(self, img_size, ratio=0.95):
        self.transformations = transforms.Compose([
            transforms.RandomCrop(size=[int(img_size[1] * ratio), int(img_size[0] * ratio)]),
            transforms.Resize((img_size[1], img_size[0]), antialias=True),
            transforms.RandomRotation(degrees=[-5.0, 5.0], expand=False),
            transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5) #, hue=0.08)
        ])
    
    def __call__(self, sample):
        sample['image'] = sample['image']/255.0
        # sample['image'] = self.transformations(sample['image'])
        return sample
    

# def get_data_processor(train_dataset, args, model_components):
#     img_size = eval(args.image_size_primary)
#     return DataTransform(img_size, 0.95)

