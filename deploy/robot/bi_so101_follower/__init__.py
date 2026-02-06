from .robot import BiSo101Follower

def obs2meta(obs):
    return BiSo101Follower.obs2meta(obs)

__all__ = ['BiSo101Follower']