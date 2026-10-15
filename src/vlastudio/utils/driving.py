"""Coordinate and camera conventions shared by Bench2Drive training and inference."""
import numpy as np

COMMANDS = ['turn left', 'turn right', 'go straight', 'follow the lane',
            'change to the left lane', 'change to the right lane']


def action_table():
    values = [(0., 0., 1., False)]
    values += [(.7, angle, 0., False) for angle in [-.5, -.3, -.2, -.1, 0., .1, .2, .3, .5]]
    values += [(.3, angle, 0., False) for angle in [-.7, -.5, -.3, -.2, -.1, 0., .1, .2, .3, .5, .7]]
    values += [(0., angle, 0., False) for angle in [-1., -.6, -.3, -.1]]
    values += [(1., 0., 0., False)]
    values += [(0., angle, 0., False) for angle in [.1, .3, .6, 1.]]
    values += [(.5, angle, 0., True) for angle in [-.5, -.3, -.2, -.1, 0., .1, .2, .3, .5]]
    return values


def command_index(value):
    value = int(value)
    if value < 0:
        value = 4
    if value not in range(1, 7):
        raise ValueError(f'Unknown CARLA route command: {value}')
    return value-1


def local_points(points, position, compass):
    theta = (float(compass) if np.isfinite(compass) else 0.)-np.pi/2
    rotation = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
    return (np.asarray(points)-np.asarray(position))@rotation.T


def observation_state(speed, target, command):
    one_hot = np.eye(6, dtype=np.float32)[command_index(command)]
    return np.concatenate([[speed], target, one_hot]).astype(np.float32)


def compose_images(images, *, jpeg=False):
    """Reproduce the authors' left/front/right crop and 256x900 panorama."""
    import torch
    left, front, right = images
    if jpeg:
        import cv2
        compressed = []
        for image in [left, front, right]:
            ok, encoded = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 20])
            if not ok:
                raise RuntimeError('CARLA sensor JPEG conversion failed')
            compressed.append(cv2.imdecode(encoded, cv2.IMREAD_COLOR))
        left, front, right = compressed
    joined = np.concatenate([left[:, :1400], front[:, 200:1400], right[:, 200:]], axis=1)
    tensor = torch.as_tensor(joined.copy()).permute(2, 0, 1)[None].float()
    resized = torch.nn.functional.interpolate(tensor, (256, 900), mode='bilinear', align_corners=False)
    return resized[0].byte().numpy()
