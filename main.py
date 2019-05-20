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
from skimage import io
import argparse
import json
import os
import models

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--image_size', type=int, default=28)
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--train', action='store_true')
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


def unnormalize(inputs, mean=0.5, std=0.5):
    return inputs * std + mean


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
    print(f'config: {config}')

    torch.manual_seed(0)
    torch.cuda.set_device(config.local_rank)

    generator = models.Generator(
        linear_params=[
            Dict(in_features=44, out_features=32),
            *[Dict(in_features=32, out_features=32)] * 64,
            Dict(in_features=32, out_features=1)
        ]
    ).cuda()
    discriminator = models.Discriminator(
        conv_params=[
            Dict(in_channels=1, out_channels=32, kernel_size=3, stride=2, bias=False),
            Dict(in_channels=32, out_channels=64, kernel_size=3, stride=2, bias=False)
        ],
        linear_param=Dict(in_features=64, out_features=10)
    ).cuda()

    generator_optimizer = torch.optim.Adam(
        params=generator.parameters(),
        lr=config.generator_lr,
        betas=(config.generator_beta1, config.generator_beta2)
    )
    discriminator_optimizer = torch.optim.Adam(
        params=discriminator.parameters(),
        lr=config.discriminator_lr,
        betas=(config.discriminator_beta1, config.discriminator_beta2)
    )

    [generator, discriminator], [generator_optimizer, discriminator_optimizer] = amp.initialize(
        models=[generator, discriminator],
        optimizers=[generator_optimizer, discriminator_optimizer],
        opt_level=config.opt_level
    )

    generator = parallel.DistributedDataParallel(generator, delay_allreduce=True)
    discriminator = parallel.DistributedDataParallel(discriminator, delay_allreduce=True)

    last_epoch = -1
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint), map_location=lambda storage, location: storage.cuda(config.local_rank))
        generator.load_state_dict(checkpoint.generator_state_dict)
        generator_optimizer.load_state_dict(checkpoint.generator_optimizer_state_dict)
        discriminator.load_state_dict(checkpoint.discriminator_state_dict)
        discriminator_optimizer.load_state_dict(checkpoint.discriminator_optimizer_state_dict)
        last_epoch = checkpoint.last_epoch

    os.makedirs(config.checkpoint_directory, exist_ok=True)
    os.makedirs(config.event_directory, exist_ok=True)
    summary_writer = SummaryWriter(config.event_directory)

    if config.train:

        dataset = datasets.MNIST(
            root='mnist',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        )

        sampler = utils.data.distributed.DistributedSampler(dataset)

        data_loader = utils.data.DataLoader(
            dataset=dataset,
            batch_size=config.local_batch_size,
            num_workers=config.num_workers,
            sampler=sampler,
            pin_memory=True,
            drop_last=True
        )

        global_step = len(data_loader) * (last_epoch + 1)

        for epoch in range(last_epoch + 1, config.num_epochs):

            discriminator.train()

            for step, (real_images, real_labels) in enumerate(data_loader):

                real_images = real_images.cuda()
                real_labels = real_labels.cuda()

                labels = nn.functional.embedding(real_labels, torch.eye(10, device='cuda'))
                labels = labels.repeat(1, config.image_size ** 2).reshape(-1, 10)

                latents = torch.randn(config.local_batch_size, 32, device='cuda')
                latents = latents.repeat(1, config.image_size ** 2).reshape(-1, 32)

                y = torch.linspace(-1, 1, config.image_size, device='cuda')
                x = torch.linspace(-1, 1, config.image_size, device='cuda')
                y, x = torch.meshgrid(y, x)
                positions = torch.stack((y.reshape(-1), x.reshape(-1)), dim=1)
                positions = positions.repeat(config.local_batch_size, 1)

                fake_images = generator(torch.cat((labels, latents, positions), dim=1))
                fake_images = fake_images.reshape(-1, 1, config.image_size, config.image_size)

                real_logits = discriminator(real_images, real_labels)
                real_logits = torch.gather(real_logits, dim=1, index=real_labels.unsqueeze(-1)).squeeze(-1)

                fake_logits = discriminator(fake_images.detach(), real_labels)
                fake_logits = torch.gather(fake_logits, dim=1, index=real_labels.unsqueeze(-1)).squeeze(-1)

                discriminator_loss = torch.mean(nn.functional.softplus(-real_logits) + nn.functional.softplus(fake_logits))
                discriminator_accuracy = torch.mean(torch.eq(torch.round(torch.sigmoid(real_logits)), 1).float())

                discriminator_optimizer.zero_grad()
                with amp.scale_loss(discriminator_loss, discriminator_optimizer) as scaled_discriminator_loss:
                    scaled_discriminator_loss.backward()
                discriminator_optimizer.step()

                fake_logits = discriminator(fake_images, real_labels)
                fake_logits = torch.gather(fake_logits, dim=1, index=real_labels.unsqueeze(-1)).squeeze(-1)

                generator_loss = torch.mean(nn.functional.softplus(-fake_logits))
                generator_accuracy = torch.mean(torch.eq(torch.round(torch.sigmoid(fake_logits)), 1).float())

                generator_optimizer.zero_grad()
                with amp.scale_loss(generator_loss, generator_optimizer) as scaled_generator_loss:
                    scaled_generator_loss.backward()
                generator_optimizer.step()

                global_step += 1

                if step % 100 == 0 and config.global_rank == 0:

                    summary_writer.add_images(
                        tag='real_images',
                        img_tensor=unnormalize(real_images.repeat(1, 3, 1, 1)),
                        global_step=global_step
                    )
                    summary_writer.add_images(
                        tag='fake_images',
                        img_tensor=unnormalize(fake_images.repeat(1, 3, 1, 1)),
                        global_step=global_step
                    )
                    summary_writer.add_scalars(
                        main_tag='training',
                        tag_scalar_dict=dict(
                            generator_loss=generator_loss,
                            discriminator_loss=discriminator_loss,
                            global_step=global_step
                        )
                    )

                    print(f'[training] epoch: {epoch} step: {step} '
                          f'generator_loss: {generator_loss:.4f} generator_accuracy: {generator_accuracy:.4f} '
                          f'discriminator_loss: {discriminator_loss:.4f} discriminator_accuracy: {discriminator_accuracy:.4f}')

            torch.save(dict(
                generator_state_dict=generator.state_dict(),
                generator_optimizer_state_dict=generator_optimizer.state_dict(),
                discriminator_state_dict=discriminator.state_dict(),
                discriminator_optimizer_state_dict=discriminator_optimizer.state_dict(),
                last_epoch=epoch
            ), f'{config.checkpoint_directory}/epoch_{epoch}')

    if config.generate:

        with torch.no_grad():

            labels = torch.multinomial(torch.ones(config.local_batch_size, 10, device='cuda'), num_samples=1).squeeze(1)
            labels = nn.functional.embedding(labels, torch.eye(10, device='cuda'))
            labels = labels.repeat(1, config.image_size ** 2).reshape(-1, 10)

            latents = torch.randn(config.local_batch_size, 32, device='cuda')
            latents = latents.repeat(1, config.image_size ** 2).reshape(-1, 32)

            y = torch.linspace(-1, 1, config.image_size, device='cuda')
            x = torch.linspace(-1, 1, config.image_size, device='cuda')
            y, x = torch.meshgrid(y, x)
            positions = torch.stack((y.reshape(-1), x.reshape(-1)), dim=1)
            positions = positions.repeat(config.local_batch_size, 1)

            images = generator(torch.cat((labels, latents, positions), dim=1))
            images = images.reshape(-1, config.image_size, config.image_size)

        for i, image in enumerate(images.cpu().numpy()):
            io.imsave(f"samples/{i}.jpg", image)

    summary_writer.close()


if __name__ == '__main__':
    main()
