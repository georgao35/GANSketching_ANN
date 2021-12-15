import jittor as jt

def group_conv_transpose(input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
    x = input
    N,C,H,W = x.shape
    i,o,h,w = weight.shape
    assert C==i
    # assert groups==1, "Group conv not supported yet."
    stride = stride if isinstance(stride, tuple) else (stride, stride)
    dilation = dilation if isinstance(dilation, tuple) else (dilation, dilation)
    # added
    padding = padding if isinstance(padding, tuple) else (padding, padding)
    output_padding = output_padding if isinstance (output_padding, tuple) else (output_padding, output_padding)
    assert output_padding[0] < max(stride[0], dilation[0]) and \
        output_padding[1] < max(stride[1], dilation[1]), \
        "output padding must be smaller than max(stride, dilation)"

    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation

    h_out = (H-1) * stride_h + output_padding[0] - 2*padding_h + 1 + (h-1)*dilation_h
    w_out = (W-1) * stride_w + output_padding[1] - 2*padding_w + 1 + (w-1)*dilation_w
    out_shape = (N, o, h_out, w_out)
    shape = (N, i, o, H, W, h, w)
    return jt.cudnn.ops.cudnn_conv_backward_x(weight, input, h_out, w_out, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, groups)
