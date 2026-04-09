"""Video batch metadata for reasoning dict – kept separate to avoid circular imports."""


def build_video_meta(num_views, horizon, is_pad, input_steps, freq):
    """Construct the ``reasoning["video"]`` metadata dict.

    Parameters
    ----------
    num_views : int   – camera views per timestep (K)
    horizon   : int   – temporal frame count (T)
    is_pad    : list[bool] – length-T, True if frame is beyond episode end
    input_steps : int – number of leading conditioning frames (reserved)
    freq      : float – sampling frequency (Hz)
    """
    return {
        "num_views": int(num_views),
        "horizon": int(horizon),
        "is_pad": is_pad,
        "freq": float(freq),
    }
