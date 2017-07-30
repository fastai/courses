import torch
import torch.nn as nn
import torch.nn.parallel

class DCGAN_D(nn.Module):
    def conv_block(self, main, name, inf, of, a, b, c, bn=True):
        main.add_module(f'{name}-{inf}.{of}.conv', nn.Conv2d(inf, of, a, b, c, bias=False))
        main.add_module(f'{name}-{of}.batchnorm', nn.BatchNorm2d(of))
        main.add_module(f'{name}-{of}.relu', nn.LeakyReLU(0.2, inplace=True))

    def __init__(self, isize, nc, ndf, ngpu, n_extra_layers=0):
        super(DCGAN_D, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        self.conv_block(main, 'initial', nc, ndf, 4, 2, 1, False)
        csize, cndf = isize / 2, ndf

        for t in range(n_extra_layers):
            self.conv_block(main, f'extra-{t}', cndf, cndf, 3, 1, 1)

        while csize > 4:
            self.conv_block(main, 'pyramid', cndf, cndf*2, 4, 2, 1)
            cndf *= 2; csize /= 2

        # state size. K x 4 x 4
        main.add_module(f'final.{cndf}-1.conv', nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        output = nn.parallel.data_parallel(self.main, input, gpu_ids)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G(nn.Module):
    def deconv_block(self, main, name, inf, of, a, b, c, bn=True):
        main.add_module(f'{name}-{inf}.{of}.convt', nn.ConvTranspose2d(inf, of, a, b, c, bias=False))
        main.add_module(f'{name}-{of}.batchnorm', nn.BatchNorm2d(of))
        main.add_module(f'{name}-{of}.relu', nn.ReLU(inplace=True))

    def __init__(self, isize, nz, nc, ngf, ngpu, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        self.ngpu = ngpu
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize: cngf *= 2; tisize *= 2

        main = nn.Sequential()
        self.deconv_block(main, 'initial', nz, cngf, 4, 1, 0)

        csize, cndf = 4, cngf
        while csize < isize//2:
            self.deconv_block(main, 'pyramid', cngf, cngf//2, 4, 2, 1)
            cngf //= 2; csize *= 2

        for t in range(n_extra_layers):
            self.deconv_block(main, f'extra-{t}', cngf, cngf, 3, 1, 1)

        main.add_module(f'final.{cngf}-{nc}.convt', nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module(f'final.{nc}.tanh', nn.Tanh())
        self.main = main

    def forward(self, input):
        gpu_ids = None
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            gpu_ids = range(self.ngpu)
        return nn.parallel.data_parallel(self.main, input, gpu_ids)

