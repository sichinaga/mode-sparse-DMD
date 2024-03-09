import os
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/6.1.1_2/bin/ffmpeg"

def error(actual, truth):
    """
    Compute relative error.
    """
    return np.linalg.norm(actual - truth) / np.linalg.norm(truth)

def hard_threshold(X, gamma):
    """
    Hard thresholding for L0 norm.
    """
    X[np.abs(X) ** 2 < 2 * gamma] = 0.0
    return X

def soft_threshold(X, gamma):
    """
    Soft thresholding for L1 norm.
    """
    return np.sign(X) * np.maximum(np.abs(X) - gamma, 0.0)

def scaled_hard_threshold(X, gamma, alpha, beta):
    """
    Scaled hard thresholding for L0 norm and L2 norm.
    """
    scale = 1 / (1 + (2 * gamma * beta))
    X = hard_threshold(X, (gamma * alpha) / scale)
    return X * scale

def scaled_soft_threshold(X, gamma, alpha, beta):
    """
    Scaled soft thresholding for L1 norm and L2 norm.
    """
    scale = 1 / (1 + (2 * gamma * beta))
    X = soft_threshold(X, gamma * alpha)
    return X * scale

def make_video_2D(
    X: np.ndarray,
    T: float,
    nx: int,
    ny: int,
    filename: str,
    scale: float = 1.0,
    order: str = "F",
    cmap: str = "viridis",
    figsize: tuple = None,
    dpi: int = None,
):
    """
    Function that converts 2-D data into a .mp4 video.

    X: 2-D matrix that contains pixel data for the video, with each
        column of X containing a single, flattened video snapshot.
    T: video duration in seconds.
    nx: number of horizontal pixels per frame.
    ny: number of vertical pixels per frame.
    filename: desired path and name for the resulting .mp4 file.
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise ValueError("Data matrix must be a 2-D numpy array.")

    X = X.real
    fps = X.shape[1] / T
    vmax = scale * np.abs(X).max()

    def make_frame(ti):
        frame = X[:, int(fps * ti)].reshape(nx, ny, order=order)
        ax.clear()
        ax.imshow(frame, vmin=-vmax, vmax=vmax, cmap=cmap)
        return mplfig_to_npimage(fig)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    animation = VideoClip(make_frame, duration=T)
    animation.write_videofile(f"{filename}.mp4", fps=fps)
