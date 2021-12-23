import jittor as jt
from jittor import init
from jittor import nn
import jittor


class GANLoss(nn.Module):
    def __init__(
        self,
        gan_mode,
        target_real_label=1.0,
        target_fake_label=0.0,
        tensor=None,
        opt=None,
    ):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.gan_mode = gan_mode
        self.opt = opt
        all_gan_modes = ["ls", "original", "w", "hinge", "softplus"]
        if gan_mode not in all_gan_modes:
            raise ValueError("Unexpected gan_mode {}".format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                # self.real_label_tensor = jt.array(self.real_label).float32()
                self.real_label_tensor = jt.full(1, self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                # self.fake_label_tensor = jt.array(self.fake_label).float32()
                # self.fake_label_tensor.stop_grad()
                self.fake_label_tensor = jt.full(1, self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            # self.zero_tensor = jt.array(1).float32()
            # self.zero_tensor.stop_grad()
            self.zero_tensor = jt.full(1, 0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == "original":
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = nn.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == "ls":
            target_tensor = self.get_target_tensor(input, target_is_real)
            return nn.mse_loss(input, target_tensor)
        elif self.gan_mode == "hinge":
            if for_discriminator:
                if target_is_real:
                    minval = jt.min((input - 1), self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
                else:
                    minval = jt.min(((-input) - 1), self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                loss = -jt.mean(input)
            return loss
        elif self.gan_mode == "softplus":
            if for_discriminator:
                if target_is_real:
                    loss = jt.mean(nn.softplus((-input)))
                else:
                    loss = jt.mean(nn.softplus(input))
            else:
                assert (
                    target_is_real
                ), "The generator's hinge loss must be aiming for real"
                loss = jt.mean(nn.softplus((-input)))
            return loss
        elif target_is_real:
            return -jt.mean(input)
        else:
            return jt.mean(input)

    def execute(self, input, target_is_real, for_discriminator=True):
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[(-1)]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if (len(loss_tensor.shape) == 0) else loss_tensor.shape[0]
                new_loss = jt.mean(loss_tensor.view((bs, (-1))), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


class WeightLoss(nn.Module):
    def __init__(self, params):
        super(WeightLoss, self).__init__()
        self.ref_weights = [p.stop_grad() for p in params]

    def execute(self, params):
        losses = []
        for i in range(len(params)):
            losses.append((params[i] - self.ref_weights[i]).abs().mean())
        loss = sum(losses) / len(losses)
        return loss


class RegularizeD(nn.Module):
    def execute(self, real_pred, real_img):
        outputs = real_pred.reshape(real_pred.shape[0], -1).mean(1).sum()
        grad_real = jittor.grad(loss=outputs, targets=real_img)
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()
        return grad_penalty
