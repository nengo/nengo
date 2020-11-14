"""
Vendorized from https://github.com/renmengye/np-conv2d

MIT License

Copyright (c) 2017 Mengye Ren

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import division

import numpy as np

from nengo.utils.numpy import array_offset


def calc_pad(pad, in_siz, out_siz, stride, ksize):
    """Calculate padding width.
    Args:
        pad: padding method, "SAME", "VALID", or manually speicified.
        ksize: kernel size [I, J].
    Returns:
        pad_: Actual padding width.
    """
    if pad == 'SAME':
        return (out_siz - 1) * stride + ksize - in_siz
    elif pad == 'VALID':
        return 0
    else:
        return pad


def calc_size(h, kh, pad, sh):
    """Calculate output image size on one dimension.
    Args:
        h: input image size.
        kh: kernel size.
        pad: padding strategy.
        sh: stride.
    Returns:
        s: output size.
    """

    if pad == 'VALID':
        return np.ceil((h - kh + 1) / sh)
    elif pad == 'SAME':
        return np.ceil(h / sh)
    else:
        return int(np.ceil((h - kh + pad + 1) / sh))


def extract_sliding_windows(x, ksize, pad, stride, floor_first=True):
    """Converts a tensor to sliding windows.
    Args:
        x: [N, H, W, C]
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]
    Returns:
        y: [N, (H-KH+PH+1)/SH, (W-KW+PW+1)/SW, KH * KW, C]
    """
    n = x.shape[0]
    h = x.shape[1]
    w = x.shape[2]
    c = x.shape[3]
    kh = ksize[0]
    kw = ksize[1]
    sh = stride[0]
    sw = stride[1]

    h2 = int(calc_size(h, kh, pad, sh))
    w2 = int(calc_size(w, kw, pad, sw))
    ph = int(calc_pad(pad, h, h2, sh, kh))
    pw = int(calc_pad(pad, w, w2, sw, kw))

    ph0 = int(np.floor(ph / 2))
    ph1 = int(np.ceil(ph / 2))
    pw0 = int(np.floor(pw / 2))
    pw1 = int(np.ceil(pw / 2))

    if floor_first:
        pph = (ph0, ph1)
        ppw = (pw0, pw1)
    else:
        pph = (ph1, ph0)
        ppw = (pw1, pw0)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0,))

    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, sh*x_sh, sw*x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c), dtype=x.dtype, buffer=x.data,
                   offset=array_offset(x), strides=y_strides)
    return y


def conv2d(x, w, pad='SAME', stride=(1, 1)):
    """2D convolution (technically speaking, correlation).
    Args:
        x: [N, H, W, C]
        w: [I, J, C, K]
        pad: [PH, PW]
        stride: [SH, SW]
    Returns:
        y: [N, H', W', K]
    """
    ksize = w.shape[:2]
    x = extract_sliding_windows(x, ksize, pad, stride)
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    xs = x.shape
    x = x.reshape([xs[0] * xs[1] * xs[2], -1])
    y = x.dot(w)
    y = y.reshape([xs[0], xs[1], xs[2], -1])
    return y
