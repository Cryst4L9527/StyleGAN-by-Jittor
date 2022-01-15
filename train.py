import jittor as jt
import numpy as np
from model import StyledGenerator, Discriminator
from dataset import SymbolDataset
from tqdm import tqdm
import argparse
import math
import random

jt.flags.use_cuda = True
jt.flags.log_silent = True
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].update(par1[k] * decay + (1 - decay) * par2[k].detach())
def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult



if __name__ == '__main__':
    path="/home/user/Desktop/stylegan/data/symbol"
    ckpt=None
    code_size=512
    batch_size={4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    lr = 1e-3
    init_size=8
    max_size=128
    use_loss='wgan-gp'
    from_rgb_act=True
    mixing = True
    sched=False
    generator=StyledGenerator(code_dim=code_size)
    discriminator=Discriminator(from_rgb_activate=from_rgb_act)
    g_running = StyledGenerator(code_size)
    g_optimizer = jt.optim.Adam(generator.generator.parameters(), lr=lr, betas=(0.0, 0.99))
    g_optimizer.add_param_group({
        'params': generator.style.parameters(),
        'lr': lr * 0.01,
        'mult': 0.01,
        }
    )
    d_optimizer = jt.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))
    accumulate(g_running, generator, 0)
    if ckpt is not None:
            ckpt = jt.load(ckpt)
            generator.load_state_dict(ckpt['generator'])
            discriminator.load_state_dict(ckpt['discriminator'])
            g_running.load_state_dict(ckpt['g_running'])
            print('from checkpoint .......')
    if sched:
        lr={128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        batch_size={4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 16}
    else:
        lr={}
        batch_size={}
    gen_sample={512: (8, 4), 1024: (4, 2)}
    batch_default=32
    phase = 200_000
    max_iter = 300_000
    
    ## Actual Training     
    step = int(math.log2(init_size) - 2)
    resolution=int(4 * 2**step)
    image_loader = SymbolDataset(path,resolution).set_attrs(
        batch_size=batch_size.get(resolution, batch_default), 
        shuffle=True
    )
    data_loader = iter(image_loader)
    adjust_lr(g_optimizer, lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, lr.get(resolution, 0.001))
    pbar = tqdm(range(max_iter))
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0
    max_step  = int(math.log2(max_size) - 2)
    final_progress = False

    for i in pbar:
        alpha = min(1, 1 / phase * (used_sample + 1))
        if (resolution == init_size and ckpt is None) or final_progress:
            alpha = 1
        
        if used_sample > phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1
            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            image_loader = SymbolDataset(path,resolution).set_attrs(
                batch_size=batch_size.get(resolution, batch_default), 
                shuffle=True
            )
            data_loader = iter(image_loader)

            jt.save(
                {
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )
            adjust_lr(g_optimizer, lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)
        except (OSError, StopIteration):
            data_loader = iter(image_loader)
            real_image = next(data_loader)
        used_sample += real_image.shape[0]
        b_size = real_image.size(0)
        if mixing and random.random() < 0.9:
                gen_in11, gen_in12, gen_in21, gen_in22 = jt.randn(4, b_size, code_size).chunk(4, 0)
                gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
                gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]
        else:
                gen_in1, gen_in2 = jt.randn(2, b_size, code_size).chunk(2, 0)
                gen_in1 = gen_in1.squeeze(0)
                gen_in2 = gen_in2.squeeze(0)

        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)
        if use_loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = -real_predict.mean() + 0.001 * (real_predict ** 2).mean()
            fake_predict = fake_predict.mean()
            eps = jt.randn(b_size, 1, 1, 1)
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = jt.grad(hat_predict.sum(),x_hat)
            grad_penalty = (
                (grad_x_hat.reshape(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
            if i % 10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()
            loss_D = real_predict+grad_penalty+fake_predict
            d_optimizer.step(loss_D)

        elif use_loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = jt.nn.softplus(-real_scores).mean()
            grad_real = jt.grad(real_scores.sum(), real_image)
            grad_penalty = (
                grad_real.reshape(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            if i % 10 == 0:
                grad_loss_val = grad_penalty.item()
            fake_predict = jt.nn.softplus(fake_predict).mean()
            if i % 10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()
            loss_D = real_predict + grad_penalty + fake_predict
            d_optimizer.step(loss_D)

        # optimize generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        fake_image = generator(gen_in2, step=step, alpha=alpha)
        predict = discriminator(fake_image, step=step, alpha=alpha)
        if use_loss == 'wgan-gp':
            loss_G = -predict.mean()
        elif use_loss == 'r1':
            loss_G = jt.nn.softplus(-predict).mean()

        if i % 10 == 0:
            gen_loss_val = loss_G.item()
        g_optimizer.step(loss_G)
        accumulate(g_running, generator)
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        if (i + 1) % 100 == 0:
            images = []

            gen_i, gen_j = gen_sample.get(resolution, (10, 5))

            with jt.no_grad():
                for _ in range(gen_i):
                    images.append(
                        g_running(
                            jt.randn(gen_j, code_size), step=step, alpha=alpha
                        ).data
                    )
            jt.save_image(
                jt.concat(images, 0),
                f'sample/{str(i + 1).zfill(6)}.png',
                nrow=gen_i,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 100 == 0:
            jt.save({'g_running':g_running.state_dict(),'step':step}, f'checkpoint/{str(i + 1).zfill(6)}.model')

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )
        pbar.set_description(state_msg)
