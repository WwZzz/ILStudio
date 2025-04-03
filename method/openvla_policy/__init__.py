import math
from method import MetaAgent
import json
import os
import time
import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from utils.prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from utils.prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from utils.prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
DEFAULT_MODEL_DICT = {
    'pretrained_checkpoint': 'openvla/openvla-7b-finetuned-libero-spatial',
    'attn_implementation':"flash_attention_2",
    'torch_dtype':torch.bfloat16,
    'load_in_8bit':False ,
    'load_in_4bit':False ,
    'low_cpu_mem_usage':True,
    'trust_remote_code':True,
}
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
    """Generates an action with the VLA policy."""
    image = Image.fromarray(obs["full_image"])
    image = image.convert("RGB")

    # (If trained with image augmentations) Center crop image and then resize back up to original size.
    # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
    #            the original height and width by sqrt(0.9) -- not 0.9!
    if center_crop:
        batch_size = 1
        crop_scale = 0.9

        # Convert to TF Tensor and record original data type (should be tf.uint8)
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype

        # Convert to data type tf.float32 and values between [0,1]
        image = tf.image.convert_image_dtype(image, tf.float32)

        # Crop and then resize back to original size
        image = crop_and_resize(image, crop_scale, batch_size)

        # Convert back to original data type
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

        # Convert back to PIL Image
        image = Image.fromarray(image.numpy())
        image = image.convert("RGB")

    # Build VLA prompt
    if "openvla-v01" in base_vla_name:  # OpenVLA v0.1
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:  # OpenVLA
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process inputs.
    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    # Get action.
    action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
    return action

class OpenVLAAgent(MetaAgent):
    def __init__(self, model_config:dict={}, task_name='libero_spatial', image_resize_size:int=224, device='cuda', center_crop=True):
        # super().__init__()
        self.model_config = model_config if model_config=={} else DEFAULT_MODEL_DICT
        assert 'pretrained_checkpoint' in self.model_config.keys()
        self.image_resize_size = image_resize_size
        self.center_crop = center_crop
        self.device = device
        self.task_name = task_name
        self.unnorm_key = task_name
        self.model = self.get_vla(**self.model_config)
        if self.unnorm_key not in self.model.norm_stats and f"{self.unnorm_key}_no_noops" in self.model.norm_stats:
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        assert self.unnorm_key in self.model.norm_stats, f"Action un-norm key {self.unnorm_key} not found in VLA `norm_stats`!"
        self.processor = AutoProcessor.from_pretrained(self.model_config['pretrained_checkpoint'], trust_remote_code=True)

    def act(self, data):
        obs, task_prompt = data['obs'], data['task_prompt']
        action = get_vla_action(
            self.model,
            self.processor, self.model_config['pretrained_checkpoint'], obs, task_prompt, self.unnorm_key, center_crop=self.center_crop
        )
        action = self.normalize_gripper_action(action, binarize=True)
        if 'libero' in self.task_name: action[..., -1] = action[..., -1] * -1.0
        return action

    def normalize_gripper_action(self, action, binarize=True):
        """
        Changes gripper action (last dimension of action vector) from [0,1] to [-1,+1].
        Necessary for some environments (not Bridge) because the dataset wrapper standardizes gripper actions to [0,1].
        Note that unlike the other action dimensions, the gripper action is not normalized to [-1,+1] by default by
        the dataset wrapper.

        Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1
        """
        # Just normalize the last action to [-1,+1].
        orig_low, orig_high = 0.0, 1.0
        action[..., -1] = 2 * (action[..., -1] - orig_low) / (orig_high - orig_low) - 1
        if binarize:
            # Binarize to -1 or +1.
            action[..., -1] = np.sign(action[..., -1])
        return action


    def resize_image(self, img, resize_size):
        """
        Takes numpy array corresponding to a single image and returns resized image as numpy array.

        NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                        the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
        """
        assert isinstance(resize_size, tuple)
        # Resize to image size expected by model
        img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        img = img.numpy()
        return img

    def process_observation(self, obs):
        if 'libero' in self.task_name:
            img = obs["agentview_image"]
            img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = self.resize_image(img, (self.image_resize_size, self.image_resize_size))
            # Prepare observations dict
            # Note: OpenVLA does not take proprio state as input
            observation = {
                "full_image": img,
                "state": np.concatenate(
                    (obs["robot0_eef_pos"], self.quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
                ),
            }
            return observation
        else:
            return obs

    def quat2axisangle(self, quat):
        """
        Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

        Converts quaternion to axis-angle format.
        Returns a unit vector direction scaled by its angle in radians.

        Args:
            quat (np.array): (x,y,z,w) vec4 float angles

        Returns:
            np.array: (ax,ay,az) axis-angle exponential coordinates
        """
        # clip quaternion
        if quat[3] > 1.0:
            quat[3] = 1.0
        elif quat[3] < -1.0:
            quat[3] = -1.0

        den = np.sqrt(1.0 - quat[3] * quat[3])
        if math.isclose(den, 0.0):
            # This is (close to) a zero degree rotation, immediately return
            return np.zeros(3)
        return (quat[:3] * 2.0 * math.acos(quat[3])) / den


    def get_vla(self,
                pretrained_checkpoint:str,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                load_in_4bit=False,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
        ):
        """Loads and returns a VLA model from checkpoint."""
        # Load VLA checkpoint.
        print("[*] Instantiating Pretrained VLA model")
        print("[*] Loading in BF16 with Flash-Attention Enabled")
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        vla = AutoModelForVision2Seq.from_pretrained(
            pretrained_checkpoint,
            attn_implementation = attn_implementation,
            torch_dtype = torch_dtype,
            load_in_8bit = load_in_8bit,
            load_in_4bit = load_in_4bit,
            low_cpu_mem_usage = low_cpu_mem_usage,
            trust_remote_code = trust_remote_code,
        )
        if not load_in_8bit and not load_in_4bit:
            vla = vla.to(self.device)
        # Load dataset stats used during finetuning (for action un-normalization).
        dataset_statistics_path = os.path.join(pretrained_checkpoint, "dataset_statistics.json")
        if os.path.isfile(dataset_statistics_path):
            with open(dataset_statistics_path, "r") as f:
                norm_stats = json.load(f)
            vla.norm_stats = norm_stats
        else:
            print(
                "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
                "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
                "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
            )
        return vla



