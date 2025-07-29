from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
from qwen2_vla.model_load_utils import load_model_for_eval
from torchvision import transforms
import pickle
from tianshou.env import SubprocVectorEnv
import time
import copy
import json
from data_utils.utils import set_seed
from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
import tensorflow as tf
from transformers.deepspeed import deepspeed_load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from typing import List
from pathlib import Path
import argparse
from collections import deque
import imageio

def add_title_caption(
    src_img,
    title: str,
    caption: str,
    *,
    title_font_path = None,   # TrueType 路径，None=自动找 DejaVuSans.ttf
    caption_font_path = None,
    title_font_size: int = 60,
    caption_font_size: int = 40,
    margin_x: int = 40,        # 画布左右留白
    margin_y: int = 25,        # 画布上下留白
    gap_title_img: int = 15,   # 标题与原图间距
    gap_img_cap: int = 15,     # 原图与注释间距
    bg_color: str = "white",
    text_color: str = "black",
):
    """
    在 src_img 的上方写标题，在下方写注释，返回新的 Image；src_img 不会被修改。
    """
    # -------------------------
    # 1. 载入字体
    # -------------------------
    def _load_font(path, size: int) -> ImageFont.FreeTypeFont:
        if path is None:
            # Pillow 自带 DejaVuSans.ttf，基本能覆盖中英文；若找不到再用默认字体
            pil_font_dir = Path(Image.__file__).with_name("fonts")
            path = pil_font_dir / "DejaVuSans.ttf"
            if not path.exists():
                return ImageFont.load_default()
        return ImageFont.truetype(str(path), size)

    font_title   = _load_font(title_font_path,   title_font_size)
    font_caption = _load_font(caption_font_path, caption_font_size)
    img = src_img.copy() 
    w_img, h_img = img.size
    tmp_draw = ImageDraw.Draw(img)
    def _text_size(text, font):
        # textbbox 兼容 Pillow >=8.0；fallback 到 textsize
        try:
            left, top, right, bottom = tmp_draw.textbbox((0, 0), text, font=font)
            return right - left, bottom - top
        except AttributeError:
            return tmp_draw.textsize(text, font=font)

    w_title,   h_title   = _text_size(title,   font_title)
    w_caption, h_caption = _text_size(caption, font_caption)
    new_w = max(w_img, w_title, w_caption) + 2 * margin_x
    new_h = (margin_y + h_title + gap_title_img +
             h_img        + gap_img_cap   +
             h_caption    + margin_y)
    canvas = Image.new("RGB", (new_w, new_h), bg_color)
    draw   = ImageDraw.Draw(canvas)
    x_title = (new_w - w_title) // 2
    y_title = margin_y
    draw.text((x_title, y_title), title, fill=text_color, font=font_title)
    x_img = (new_w - w_img) // 2
    y_img = y_title + h_title + gap_title_img
    canvas.paste(img, (x_img, y_img))
    x_cap = (new_w - w_caption) // 2
    y_cap = y_img + h_img + gap_img_cap
    draw.text((x_cap, y_cap), caption, fill=text_color, font=font_caption)
    return canvas

def draw_point_on_image(image, xy, color='red', edge_color='white', radius=4, border=1, draw=None, use_copy=True):
    if use_copy: image=copy.deepcopy(image)
    px, py = xy
    Rt = radius + border
    if draw is None: draw = ImageDraw.Draw(image)
    draw.ellipse((px - Rt, py - Rt, px + Rt, py + Rt), fill=edge_color)
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)
    return image    

def draw_output_on_image(img: Image, task:str, text:str):
    tlist = text.split('\n')[:4]
    tlist = [ti.split(':')[1].strip() for ti in tlist]
    subtask = tlist[0]
    gpos = np.array(eval(tlist[1]))
    tpos = np.array(eval(tlist[2]))
    direction = tlist[3][:16]
    img_size = np.array(img.size)
    gpos = gpos*img_size
    tpos = tpos*img_size
    img_w_gpos = draw_point_on_image(img, gpos)
    img_gtpos = draw_point_on_image(img_w_gpos, tpos, color='yellow')
    img_title = add_title_caption(img_gtpos, task, subtask+' | '+direction)
    return img_title

def save_images_as_gif(images: List[Image.Image], filename: str, duration: int = 100, loop: int = 0) -> None:
    """
    将一个 PIL.Image 对象的列表保存为 GIF 动画。

    Args:
        images (List[Image.Image]): 包含 PIL.Image 对象的列表。
        filename (str): 要保存的 GIF 文件的路径和名称 (例如 'animation.gif')。
        duration (int, optional): 每一帧的显示时间（单位：毫秒）。默认为 100。
        loop (int, optional): GIF 动画的循环次数。0 表示无限循环。默认为 0。

    Returns:
        None: 函数执行完毕后不返回任何内容，但会生成文件。
    """
    # 检查输入的图像列表是否为空
    if not images:
        raise ValueError("输入的图像列表不能为空！")

    # 确保保存路径的目录存在
    save_dir = os.path.dirname(filename)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 以第一张图为基础，将后续的图片作为帧添加进去
    images[0].save(
        filename,
        save_all=True,           # 核心参数，必须为 True
        append_images=images[1:],# 附加从第二张开始的所有图像
        duration=duration,       # 设置每帧的持续时间
        loop=loop,               # 设置循环次数
        optimize=False           # 可选：是否优化调色板，False保证质量
    )
    print(f"GIF 已成功保存到: {filename}")
    
def resize_image(img, resize_size):
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

def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img

def pre_process(robot_state_value, key, stats):
    tmp = robot_state_value
    tmp = (tmp - stats[key + '_mean']) / stats[key + '_std']
    return tmp

def quat2axisangle(quat):
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

def get_state_from_obs(obs):
    xyz = obs['robot0_eef_pos']
    euler = quat2axisangle(obs['robot0_eef_quat'])
    gqpos = obs['robot0_gripper_qpos']
    gqpos = np.array([(gqpos[0]-gqpos[1])/0.08])
    return np.concatenate([xyz, euler, gqpos], axis=0)
    
def process_obs(obs, stats):
    """
    obs: three cameras' images
    states: Tensor, robot states
    stats: mean, std of robot states and actions
    This function is used to get observations(images and robot states) in your robot environment.
    """
    states = get_state_from_obs(obs)
    cur_img = get_libero_image(obs, 256)
    traj_rgb_np = np.array([cur_img]) # sequential must align with constants.py
    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))
    cur_state_np = pre_process(states, 'qpos', stats)
    cur_state = np.expand_dims(cur_state_np, axis=0)
    return traj_rgb_np, cur_state # images, states

def time_ms():
    return time.time_ns() // 1_000_000

class qwen2_vla_policy:
    def __init__(self, policy_config, data_args=None):
        super(qwen2_vla_policy).__init__()
        self.load_policy(policy_config)
        self.data_args = data_args

    def load_policy(self, policy_config):
        self.policy_config = policy_config
        model_base = policy_config["model_base"] if policy_config[
            'enable_lora'] else None
        model_path = policy_config["model_path"]
        self.tokenizer, self.policy, self.multimodal_processor, self.context_len = load_model_for_eval(model_path=model_path,
                                                                                                    model_base=model_base, policy_config=policy_config)
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["[SOA]"]})
        self.config = AutoConfig.from_pretrained('/'.join(model_path.split('/')[:-1]), trust_remote_code=True)
        # deepspeed_load_checkpoint(self.policy, policy_config['model_path'], load_module_strict=True)
        
    def datastruct_droid2qwen2vla(self, raw_lang, num_images):
        messages = [
            {
                "role": "user",
                "content": [],
            },
        ]
        for _ in range(num_images):
            messages[0]['content'].append({
                        "type": "image",
                        "image": None,
                    })
        
        messages[0]['content'].append({"type": "text", "text": raw_lang}) 
        return messages
    
    def process_batch_to_qwen2_vla(self, curr_image, robo_state, raw_lang):
        batch_size = robo_state.shape[0]
        len_images = curr_image.shape[1]
        instances = []
        for sid in range(batch_size):
            messages = self.datastruct_droid2qwen2vla(raw_lang, len_images)
            img_sid = curr_image[sid]
            image_data = torch.chunk(img_sid, img_sid.shape[0], dim=0)  # top, left_wrist, right_wrist
            image_list = []
            for i, each in enumerate(image_data):
                ele = {}
                each = Image.fromarray(each.cpu().squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8))
                ele['image'] = each
                ele['resized_height'] = 256
                ele['resized_width'] = 256
                image_list.append(torch.from_numpy(np.array(each)))
            image_data = image_list
            text = self.multimodal_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            video_inputs = None
            model_inputs = self.multimodal_processor(
                text=text,
                images=image_data,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            data_dict = dict(
                messages=messages,
                images=None,
            )
        # data_dict = dict(states=robo_state)
            for k, v in model_inputs.items():
                data_dict[k] = v
            instances.append(data_dict) 
        input_ids = [torch.flip(instance['input_ids'].squeeze(0), dims=[0]) for instance in instances]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        input_ids = torch.flip(input_ids, dims=[1])
        b = input_ids.shape[0]
        image_grid_thw = torch.stack([instance['image_grid_thw'] for instance in instances])
        pixel_values = torch.stack([instance['pixel_values'] for instance in instances])
        image_grid_thw = image_grid_thw.reshape(b * image_grid_thw.shape[1], image_grid_thw.shape[2])
        pixel_values = pixel_values.reshape(b * pixel_values.shape[1], pixel_values.shape[2])
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        states = robo_state
        batch = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            states=states,
        )
        return batch

def eval_bc(policy, env, policy_config, raw_lang=None, query_frequency=16, max_timesteps=400, video_writer=None):
    assert raw_lang is not None, "raw lang is None"
    policy.policy.eval()
    # 加载数据的stats
    stats_path = os.path.join("/".join(policy_config['model_path'].split('/')[:-1]), f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    # 根据stats定义动作后处理
    def post_process(a):
        actions = (((a + 1) / 2) * (stats['action_max'] - stats['action_min']) + stats['action_min'])
        actions[:,:6] = actions[:,:6]/np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.5])
        actions[:,6] = 1.-2.*actions[:,6]
        return actions 
    # 设置动作队列
    action_queue = deque(maxlen=query_frequency)
    num_waits = 10
    image_list = []
    reasoning_list = []
    video_frames = [[] for _ in range(len(env))]
    horizons = np.ones(len(env))*max_timesteps
    # env.reset()
    # 等待libero初始化的时间
    for r in range(num_waits):
        obs, reward, done, info = env.step([np.zeros((7,)).tolist() for _ in range(len(env))])
    # 开始测试
    with torch.inference_mode():
        time_start_eval = time.time()
        success =  np.zeros(len(env)).astype(np.bool8)
        for t in range(max_timesteps):
            # 处理batch化的obs
            img_array = []
            state_array = []
            for obs_i in obs:
                img_i, state_i = process_obs(obs_i, stats)
                img_array.append(img_i)
                state_array.append(state_i)
            image_list.append([Image.fromarray(img.squeeze(0).squeeze(0).transpose(1,2,0)) for img in img_array])
            if t % query_frequency == 0:
                robot_state = torch.from_numpy(np.concatenate(state_array, axis=0)).float().cuda()
                curr_image = torch.from_numpy(np.concatenate(img_array, axis=0)).float().cuda()
                batch = policy.process_batch_to_qwen2_vla(curr_image, robot_state, raw_lang)
                all_actions, outputs = policy.policy.evaluate(**batch, is_eval=True, tokenizer=policy.tokenizer)
                # clear previous actions
                while len(action_queue) > 0:
                    action_queue.popleft()
                action_queue.extend(torch.chunk(all_actions, chunks=all_actions.shape[1], dim=1)[0:query_frequency])
                try:
                    frames = image_list[-1]
                    for env_i in range(len(env)):
                        img = frames[env_i]
                        reason_img = draw_output_on_image(img, raw_lang, outputs[env_i][:min(140, len(outputs[env_i]))]).resize((400, 400))
                        video_frames[env_i].append(np.array(reason_img))
                except:
                    pass
            raw_action = action_queue.popleft()
            raw_action = raw_action.squeeze(1).cpu().to(dtype=torch.float32).numpy()
            action = post_process(raw_action)
            obs, reward, done, info = env.step(action)
            success = success | done
            if success.all():
                break
            if success.any():
                success_idx = np.where(success==True)[0]
                for sidx in success_idx:
                    if horizons[sidx]>t: horizons[sidx] = t
    env.close()
    total_successes = success.sum()
    total = len(env)
    success_rate = 1.0*total_successes/len(env)
    if video_writer is not None:
        for env_i in range(len(env)):
            for frame in video_frames[env_i]:
                video_writer.append_data(frame)
    return {
        'success': total_successes,
        'total': total,
        'success_rate': success_rate,
        'horizon': horizons,
    }
    
def create_env(task_suite_name="libero_object", task_id = 0):
    benchmark_dict = benchmark.get_benchmark_dict()
    # can also choose libero_spatial, libero_object, etc.
    task_suite = benchmark_dict[task_suite_name]()
    # retrieve a specific task
    task = task_suite.get_task(task_id)
    task_name = task.name
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    # step over the environment
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,
        "camera_widths": 256,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    env.reset()
    return env

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--ckpt', help='the path of the checkpoint', type=str, default='/inspire/hdd/project/robot-action/wangzheng-240308120196/DexVLA-Vision/qwen2_vla_libero_all_vision/checkpoint-160000')
    parser.add_argument('--task', help='task suite name', type=str, default='libero_10')
    parser.add_argument('--task_id', help='the task id in the suite', type=int, default=-1)
    parser.add_argument('--num_rollouts', help='Number of episodes to process', type=int, default=5)
    parser.add_argument('--num_envs', help='Number of episodes to process', type=int, default=5)
    parser.add_argument('--max_timesteps', help='the maximum time steps', type=int, default=100)
    parser.add_argument('--save_path', help='the path of the saved results', type=str, default='eval_results_test_parallel')
    parser.add_argument('--freq', help='query frequency', type=int, default=10)
    parser.add_argument('--save_res', help='whether to save gif', type=bool, default=True)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    set_seed(0)
    args = parse_args()
    # load policy
    policy_config = {
        "model_path": args.ckpt,
        "model_base": None, # only use for lora finetune
        "enable_lora": False, # only use for lora finetune
        "action_head": 'unet_diffusion_policy',
        "tinyvla": False,
    }
    policy = qwen2_vla_policy(policy_config)
    # get benchmark info
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task]()
    num_tasks_in_suite = task_suite.n_tasks
    task_dir = os.path.join(args.save_path, f"{args.task}") if args.save_res else None
    all_res = []
    num_envs = max(args.num_envs, 1)
    if args.task_id==-1:
        for tid in range(num_tasks_in_suite):
            # 存储路径
            if task_dir is not None: 
                os.makedirs(task_dir, exist_ok=True)
                video_path = os.path.join(task_dir, f"{tid:02d}") + '.mp4'
                video_writer = imageio.get_writer(video_path, fps=max(int(15/args.freq), 1))
            else: 
                video_writer = None
            # 获取环境的初始状态和语言指令
            init_states = task_suite.get_task_init_states(tid)
            raw_lang =  task_suite.get_task(args.task_id).language
            # inner loop for parallelism of rollouts
            res_tid = []
            for rid in range(0, args.num_rollouts, num_envs):
                # 初始化并行的带states的env
                batch_states = [init_states[(rid+i)%len(init_states)] for i in range(num_envs)]
                def make_env_for_state(state):
                    def _thunk():
                        env = create_env(args.task, tid)
                        env.set_init_state(state)
                        env.reset()
                        return env
                    return _thunk
                env_fns = [make_env_for_state(state) for state in batch_states]
                env = SubprocVectorEnv(env_fns)
                # 评测env
                res = eval_bc(policy, env, policy_config, raw_lang=raw_lang, query_frequency=args.freq, max_timesteps=args.max_timesteps, video_writer=video_writer)
                res_tid.append(res)
            # process res_tid
            total = 0
            total_successes = 0
            horizons = []
            for res_i in res_tid:
                total += res_i['total']
                total_successes += res_i['success']
                horizons.append(res_i['horizon'])
            res_tid = {'success_rate': 1.0*total_successes/total, 'horizon':np.concatenate(horizons).mean()}
            if task_dir is not None: 
                tid_json = os.path.join(task_dir, f'{tid:02d}.json')
                with open(tid_json, 'w') as f:
                    json.dump(res_tid, f)
            # append red_tid to all_res
            all_res.append(res_tid)
        all_sr = np.array([ri['success_rate'] for ri in all_res]).mean()
        all_horizons = np.array([ri['horizon'] for ri in all_res]).mean()
        if task_dir is not None: 
            tid_json = os.path.join(task_dir, f'{tid:02d}.json')
            with open(tid_json, 'w') as f:
                json.dump(res_tid, f)
    else:
        assert args.task_id<num_tasks_in_suite, "the task_id cannot exceed the number of tasks in the task suite"
        # init env
        init_states = task_suite.get_task_init_states(args.task_id)
        raw_lang =  task_suite.get_task(args.task_id).language
        env = create_env(args.task, args.task_id)
        # eval
        if task_dir is not None: 
            os.makedirs(task_dir, exist_ok=True)
            video_path = os.path.join(task_dir, f"{tid:02d}") + '.mp4'
            video_writer = imageio.get_writer(video_path, fps=max(int(15/args.freq), 1))
        else: 
            video_writer = None
            res_tid = []
        for rid in range(0, args.num_rollouts, num_envs):
            # 初始化并行的带states的env
            batch_states = [init_states[(rid+i)%len(init_states)] for i in range(num_envs)]
            def make_env_for_state(state):
                def _thunk():
                    env = create_env(args.task, tid)
                    env.set_init_state(state)
                    env.reset()
                    return env
                return _thunk
            env_fns = [make_env_for_state(state) for state in batch_states]
            env = SubprocVectorEnv(env_fns)
            # 评测env
            res = eval_bc(policy, env, policy_config, raw_lang=raw_lang, query_frequency=args.freq, max_timesteps=args.max_timesteps, video_writer=video_writer)
            res_tid.append(res)
        total = 0
        total_successes = 0
        horizons = []
        for res_i in res_tid:
            total += res_i['total']
            total_successes += res_i['success']
            horizons.append(res_i['horizon'])
        res_tid = {'success_rate': 1.0*total_successes/total, 'horizon':np.concatenate(horizons).mean()}
        if task_dir is not None: 
            tid_json = os.path.join(task_dir, f'{tid:02d}.json')
            with open(tid_json, 'w') as f:
                json.dump(res_tid, f)
        # append red_tid to all_res
        all_res.append(res_tid)

    # save results
    