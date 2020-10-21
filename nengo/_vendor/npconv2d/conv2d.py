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
        return max((out_siz - 1) * stride + ksize - in_siz, 0)
    elif pad == 'VALID':
        return 0
    else:
        return pad


def calc_gradx_pad(pad, in_siz, out_siz, stride, ksize):
    if pad == 'SAME':
        out_siz_min = (in_siz - 1) * stride + 1
        p = out_siz + ksize - 1 - out_siz_min
        p = max(p, 0)
        p = min(p, (ksize - 1) * 2)
        return p
    elif pad == 'VALID':
        return (ksize - 1) * 2
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
    n, h, w, c = x.shape
    kh, kw = ksize
    sh, sw = stride

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


def extract_sliding_windows_gradx(x,
                                  ksize,
                                  pad,
                                  stride,
                                  orig_size,
                                  floor_first=False):
    """Extracts windows on a dilated image.

    Args:
        x: [N, H', W', C] (usually dy)
        k: [KH, KW]
        pad: [PH, PW]
        stride: [SH, SW]
        orig_size: [H, W]

    Returns:
        y: [N, H, W, KH, KW, C]
    """
    n, h, w, c = x.shape
    kh, kw = ksize
    sh, sw = stride
    ph, pw = pad
    sh, sw = stride
    h2, w2 = orig_size

    xs = np.zeros([n, h, sh, w, sw, c])
    xs[:, :, 0, :, 0, :] = x
    xss = xs.shape
    x = xs.reshape([xss[0], xss[1] * xss[2], xss[3] * xss[4], xss[5]])
    x = x[:, :h2, :w2, :]

    ph2 = int(np.ceil(ph / 2))
    ph3 = int(np.floor(ph / 2))
    pw2 = int(np.ceil(pw / 2))
    pw3 = int(np.floor(pw / 2))
    if floor_first:
        pph = (ph3, ph2)
        ppw = (pw3, pw2)
    else:
        pph = (ph2, ph3)
        ppw = (pw2, pw3)
    x = np.pad(
        x, ((0, 0), pph, ppw, (0, 0)),
        mode='constant',
        constant_values=(0.0, ))

    # The following code extracts window without copying the data:
    # y = np.zeros([n, h2, w2, kh, kw, c])
    # for ii in range(h2):
    #     for jj in range(w2):
    #         y[:, ii, jj, :, :, :] = x[:, ii:ii + kh, jj:jj + kw, :]
    x_sn, x_sh, x_sw, x_sc = x.strides
    y_strides = (x_sn, x_sh, x_sw, x_sh, x_sw, x_sc)
    y = np.ndarray((n, h2, w2, kh, kw, c),
                   dtype=x.dtype,
                   buffer=x.data,
                   offset=array_offset(x),
                   strides=y_strides)
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


def conv2d_gradx(w, dy, xsize, pad='SAME', stride=(1, 1)):
    """2D convolution gradient wrt. input.

    Args:
        dy: [N, H', W', K]
        w: [I, J, K, C]
        xsize: Original image size, [H, W]

    Returns:
        dx: [N, H, W, C]

    Modifications from original code:
      - Do not transpose input/output channels in `w`
      - Fix bug in computing `pad2w` (use `dys[1]` instead of `dys[0]`)
      - Add new `calc_gradx_pad` function to ensure we compute padding the same as TF
    """
    assert w.shape[-2] == dy.shape[-1]

    dys = dy.shape[1:3]
    ksize = w.shape[:2]
    pad2 = (
        calc_gradx_pad(pad, dys[0], xsize[0], stride[0], ksize[0]),
        calc_gradx_pad(pad, dys[1], xsize[1], stride[1], ksize[1]),
    )

    dx = extract_sliding_windows_gradx(dy, ksize, pad2, stride, xsize)
    dxs = dx.shape
    dx = dx.reshape([dxs[0] * dxs[1] * dxs[2], -1])
    w = w[::-1, ::-1, :, :]
    ws = w.shape
    w = w.reshape([ws[0] * ws[1] * ws[2], ws[3]])
    dx = dx.dot(w)
    return dx.reshape([dxs[0], dxs[1], dxs[2], -1])
