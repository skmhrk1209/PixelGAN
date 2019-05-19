import torch
from torch import nn
from torch import distributed
from torch import optim
from torch import utils
from torch import cuda
from torch import backends
from torchvision import datasets
from torchvision import transforms
from tensorboardX import SummaryWriter
from apex import amp
from apex import parallel
import argparse
import json
import os
import models

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--image_size', type=int, default=1024)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training', action='store_true')
parser.add_argument('--generate', action='store_true')
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()

backends.cudnn.benchmark = True


class Dict(dict):

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


def apply(function, dictionary):
    if isinstance(dictionary, dict):
        for key, value in dictionary.items():
            dictionary[key] = apply(function, value)
        dictionary = function(dictionary)
    return dictionary


def main():

    distributed.init_process_group(backend='nccl')

    with open(args.config) as file:
        config = Dict(json.load(file))
    config.update(vars(args))
    config.update(dict(
        world_size=distributed.get_world_size(),
        global_rank=distributed.get_rank(),
        device_count=torch.cuda.device_count()
    ))
    config = apply(Dict, config)
    print(f"config: {config}")

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    encoder = models.VAE(
        conv_params=[
            Dict(in_channels=1, out_channels=32, kernel_size=3, stride=2),
            Dict(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        ],
        linear_params=[
            Dict(in_features=2304, out_features=128),
            Dict(in_features=128, out_features=64)
        ]
    ).cuda()

    generator = models.Generator(
        linear_params=[
            Dict(in_features=34, out_features=32),
            *[Dict(in_features=32, out_features=32)] * 128,
            Dict(in_features=32, out_features=1)
        ]
    ).cuda()

    discriminator = models.Discriminator(
        conv_params=[
            Dict(in_channels=1, out_channels=32, kernel_size=3, stride=2, bias=False),
            Dict(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=False)
        ],
        linear_params=[
            Dict(in_features=2304, out_features=128),
            Dict(in_features=128, out_features=10)
        ]
    ).cuda()

    generator_optimizer = torch.optim.Adam(
        params=encoder.parameters() + generator.parameters(),
        lr=config.generator_lr,
        betas=(config.generator_beta1, config.generator_beta2)
    )
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config.discriminator_lr,
        betas=(config.discriminator_beta1, config.discriminator_beta2)
    )

    generator, generator_optimizer = amp.initialize(generator, generator_optimizer, opt_level=config.opt_level)
    discriminator, discriminator_optimizer = amp.initialize(discriminator, discriminator_optimizer, opt_level=config.opt_level)

    generator = parallel.DistributedDataParallel(generator, delay_allreduce=True)
    discriminator = parallel.DistributedDataParallel(discriminator, delay_allreduce=True)

    last_epoch = -1
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint), map_location=lambda storage, location: storage.cuda(local_rank))
        generator.load_state_dict(checkpoint.generator_state_dict)
        generator_optimizer.load_state_dict(checkpoint.generator_optimizer_state_dict)
        discriminator.load_state_dict(checkpoint.discriminator_state_dict)
        discriminator_optimizer.load_state_dict(checkpoint.discriminator_optimizer_state_dict)
        last_epoch = checkpoint.last_epoch

    summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)

        dataset = datasets.MNIST(
            root="mnist",
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )

        sampler = utils.data.distributed.DistributedSampler(dataset)

        data_loader = utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.local_batch_size,
            num_workers=config.num_workers,
            sampler=sampler,
            pin_memory=True
        )

        for epoch in range(last_epoch + 1, config.num_epochs):

            generator.train()
            discriminator.train()

            for step, (real_images, real_labels) in enumerate(data_loader):

                real_images = real_images.cuda()
                real_labels = real_labels.cuda()

                latents, kl_divergences = encoder(real_images)

                latents = torch.randn(config.local_batch_size, 128, device='cuda')
                latents = latents.repeat(1, 1 * config.image_size ** 2).reshape(-1, 128)

                y = torch.arange(config.image_size).cuda()
                x = torch.arange(config.image_size).cuda()
                y, x = torch.meshgrid(y, x)
                positions = torch.stack((y.reshape(-1), x.reshape(-1)), dim=-1)
                positions = (positions.float() - config.image_size / 2) / (config.image_size / 2)
                positions = positions.repeat(config.local_batch_size, 1)

                fake_images = generator(torch.cat((latents, positions), dim=-1))
                fake_images = fake_images.reshape(config.local_batch_size, 1, config.image_size, config.image_size)

                real_logits = discriminator(real_images).reshape(-1, 10)
                fake_logits = discriminator(fake_images.detach()).reshape(-1, 10)

                real_logits = torch.gather(real_logits, dim=1, index=real_labels.unsqueeze(-1)).squeeze(-1)
                fake_logits = torch.gather(fake_logits, dim=1, index=real_labels.unsqueeze(-1)).squeeze(-1)

                real_losses = nn.functional.softplus(-real_logits)
                fake_losses = nn.functional.softplus(fake_logits)
                discriminator_losses = real_losses + fake_losses

                discriminator_loss = torch.mean(discriminator_losses)
                discriminator_optimizer.zero_grad()
                with amp.scale_loss(discriminator_loss, discriminator_optimizer) as scaled_discriminator_loss:
                    scaled_discriminator_loss.backward()
                discriminator_optimizer.step()

                fake_logits = discriminator(fake_images).reshape(-1, 10)
                fake_logits = torch.gather(fake_logits, dim=1, index=real_labels.unsqueeze(-1)).squeeze(-1)

                fake_losses = nn.functional.softplus(-fake_logits)
                generator_losses = fake_losses + kl_divergences * config.kl_divergence_weight

                generator_loss = torch.mean(generator_losses)
                generator_optimizer.zero_grad()
                with amp.scale_loss(generator_loss, generator_optimizer) as scaled_generator_loss:
                    scaled_generator_loss.backward()
                generator_optimizer.step()

                if step % 100 == 0 and config.global_rank == 0:
                    summary_writer.add_image(
                        tag='real_images',
                        img_tensor=real_images
                    )
                    summary_writer.add_image(
                        tag='fake_images',
                        img_tensor=fake_images
                    )
                    summary_writer.add_scalars(
                        main_tag='training',
                        tag_scalar_dict=dict(
                            generator_loss=generator_loss,
                            discriminator_loss=discriminator_loss
                        )
                    )
                    print(f'[training] epoch: {epoch} step: {step} '
                          f'generator_loss: {generator_loss} discriminator_loss: {discriminator_loss}')

            torch.save(dict(
                encoder_state_dict=encoder.state_dict(),
                generator_state_dict=generator.state_dict(),
                generator_optimizer_state_dict=generator_optimizer.state_dict(),
                discriminator_state_dict=discriminator.state_dict(),
                discriminator_optimizer_state_dict=discriminator_optimizer.state_dict(),
                last_epoch=epoch
            ), f'{config.checkpoint_directory}/epoch_{epoch}')

    summary_writer.close()


if __name__ == '__main__':
    main()
