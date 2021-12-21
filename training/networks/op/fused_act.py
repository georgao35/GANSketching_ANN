import jittor as jt
from jittor import nn
from jittor import Function, Var


def fused_bias_act_code_op(input: Var, bias: Var, refer: Var, act, grad, alpha, scale):
    use_bias = 1 if bias.numel() else 0
    use_ref = 1 if refer.numel() else 0
    step_b = 1
    for i in range(2, len(input.shape)):
        step_b *= input.shape[i]
    return jt.code(
        shape=input.shape,
        dtype=input.dtype,
        inputs=[input, bias, refer],
        cuda_header="""
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
static __global__ void fused_bias_act_kernel(scalar_t* out, const scalar_t* p_x, const scalar_t* p_b, const scalar_t* p_ref,
    int act, int grad, scalar_t alpha, scalar_t scale, int loop_x, int size_x, int step_b, int size_b, int use_bias, int use_ref) {
    int xi = blockIdx.x * loop_x * blockDim.x + threadIdx.x;

    scalar_t zero = 0.0;

    for (int loop_idx = 0; loop_idx < loop_x && xi < size_x; loop_idx++, xi += blockDim.x) {
        scalar_t x = p_x[xi];

        if (use_bias) {
            x += p_b[(xi / step_b) % size_b];
        }

        scalar_t ref = use_ref ? p_ref[xi] : zero;

        scalar_t y;

        switch (act * 10 + grad) {
            default:
            case 10: y = x; break;
            case 11: y = x; break;
            case 12: y = 0.0; break;

            case 30: y = (x > 0.0) ? x : x * alpha; break;
            case 31: y = (ref > 0.0) ? x : x * alpha; break;
            case 32: y = 0.0; break;
        }

        out[xi] = y * scale;
    }
}
        """,
        cuda_src=f"""
    @alias(input, in0)
    @alias(bias, in1)
    @alias(refer, in2)
    @alias(output, out)

    int use_bias = {use_bias};
    int use_ref = {use_ref};

    int size_x = {input.numel()};
    int size_b = {bias.numel()};
    int step_b = {step_b};

    int loop_x = 4;
    int block_size = 4 * 32;
    int grid_size = (size_x - 1) / (loop_x * block_size) + 1;

        fused_bias_act_kernel<input_type><<<grid_size, block_size>>>(
            output_p,
            input_p,
            bias_p,
            refer_p,
            {act},
            {grad},
            {alpha},
            {scale},
            loop_x,
            size_x,
            step_b,
            size_b,
            use_bias,
            use_ref
        );
""",
    )


class FusedLeakyReLUFunction(Function):
    def __init__(self) -> None:
        super().__init__()

    def execute(self, input: Var, bias: Var, negative_slope, scale) -> Var:
        empty = jt.empty(shape=(0,), dtype=input.dtype)

        self.bias = bias is not None

        if bias is None:
            bias = empty

        out = fused_bias_act_code_op(input, bias, empty, 3, 0, negative_slope, scale)
        # save for backward
        self.save_vars = out
        self.negative_slope = negative_slope
        self.scale = scale

        return out

    def grad(self, grad_output: Var):
        if grad_output is None:
            return None, None, None, None
        # print("fused_act not None")
        out = self.save_vars

        empty = jt.empty(shape=(0,), dtype=grad_output.dtype)

        grad_input = fused_bias_act_code_op(
            grad_output, empty, out, 3, 1, self.negative_slope, self.scale
        )

        dim = [0]

        if grad_input.ndim > 2:
            dim += list(range(2, grad_input.ndim))

        if self.bias:
            grad_bias = grad_input.sum(dim).detach()

        else:
            grad_bias = empty

        if not self.bias:
            grad_bias = None

        return grad_input, grad_bias, None, None


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, bias=True, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        if bias:
            self.bias = jt.zeros(channel)

        else:
            self.bias = None

        self.negative_slope = negative_slope
        self.scale = scale

    def execute(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input: Var, bias: Var = None, negative_slope=0.2, scale=2 ** 0.5):
    if jt.flags.use_cuda:
    # if False:
        return FusedLeakyReLUFunction.apply(input, bias, negative_slope, scale)
    else:
        if bias is not None:
            rest_dim = [1] * (input.ndim - bias.ndim - 1)
            return (
                nn.leaky_relu(input + bias.view(1, bias.shape[0], *rest_dim), 0.2)
                * scale
            )
        else:
            return nn.leaky_relu(input, negative_slope=0.2) * scale
