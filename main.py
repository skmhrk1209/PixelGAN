import torch
from torch import nn
from torch import distributed
from torch import optim
from torch import utils
from torch import cuda
from torch import backends
from torchvision import datasets
from torchvision import transforms
from torchvision import models
from tensorboardX import SummaryWriter
from nvidia import dali
from nvidia.dali.plugin import pytorch
from apex import amp
from apex import parallel
from pipeline import Pipeline
import argparse
import json
import os

parser = argparse.ArgumentParser(description='ResNet50 training on Imagenet')
parser.add_argument('--config', type=str, default='config.json')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--training', action='store_true')
parser.add_argument('--evaluation', action='store_true')
parser.add_argument('--inference', action='store_true')
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()

backends.cudnn.benchmark = True


class Dict(dict):

    def __getattr__(self, name): return self[name]

    def __setattr__(self, name, value): self[name] = value

    def __delattr__(self, name): del self[name]


def main():

    with open(args.config) as file:
        config = Dict(json.load(file))
        config.update(vars(args))

    distributed.init_process_group(backend='nccl')
    world_size = distributed.get_world_size()
    global_rank = distributed.get_rank()
    device_count = torch.cuda.device_count()
    local_rank = config.local_rank
    torch.cuda.set_device(local_rank)
    print(f'Enabled distributed training. (global_rank: {global_rank}/{world_size}, local_rank: {local_rank}/{device_count})')

    torch.manual_seed(0)

    generator = nn.Sequential(
        nn.Sequential(
            nn.Linear(130, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.Sigmoid()
        ),
        *[nn.Sequential(
            nn.Linear(32, 32, bias=False),
            nn.BatchNorm1d(32),
            nn.Sigmoid()
        ) for _ in range(128)],
        nn.Sequential(
            nn.Linear(32, 1, bias=False),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
    )
    generator = generator.cuda()

    discriminator = models.resnet18()
    discriminator.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    nn.init.kaiming_normal_(discriminator.conv1.weight, nonlinearity="relu")
    discriminator.fc = nn.Linear(in_features=2048, out_features=10, bias=True)
    nn.init.xavier_normal_(discriminator.fc.weight)
    nn.init.zeros_(discriminator.fc.bias)
    discriminator = discriminator.cuda()

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

    generator, generator_optimizer = amp.initialize(generator, generator_optimizer, opt_level=config.opt_level)
    generator = parallel.DistributedDataParallel(generator, delay_allreduce=True)

    discriminator, discriminator_optimizer = amp.initialize(discriminator, discriminator_optimizer, opt_level=config.opt_level)
    discriminator = parallel.DistributedDataParallel(discriminator, delay_allreduce=True)

    last_epoch = -1
    if config.checkpoint:
        checkpoint = Dict(torch.load(config.checkpoint), map_location=lambda storage, location: storage.cuda(local_rank))
        generator.load_state_dict(checkpoint.generator_state_dict)
        generator_optimizer.load_state_dict(checkpoint.generator_optimizer_state_dict)
        discriminator.load_state_dict(checkpoint.discriminator_state_dict)
        discriminator_optimizer.load_state_dict(checkpoint.discriminator_optimizer_state_dict)
        last_epoch = checkpoint.last_epoch

    criterion = nn.CrossEntropyLoss(reduction='mean').cuda()

    summary_writer = SummaryWriter(config.event_directory)

    if config.training:

        os.makedirs(config.checkpoint_directory, exist_ok=True)
        os.makedirs(config.event_directory, exist_ok=True)

        # NOTE: When partition for distributed training executed?
        # NOTE: Should random seed be the same in the same node?
        pipeline = Pipeline(
            root=config.train_root,
            train=True,
            batch_size=config.local_batch_size,
            num_threads=config.num_workers,
            device_id=local_rank,
            num_shards=world_size,
            shard_id=global_rank,
            image_size=224
        )
        pipeline.build()

        # NOTE: What's `epoch_size`?
        # NOTE: Is that len(dataset) ?
        data_loader = pytorch.DALIClassificationIterator(
            pipelines=pipeline,
            size=list(pipeline.epoch_size().values())[0] // world_size,
            auto_reset=True,
            stop_at_epoch=True
        )

        for epoch in range(last_epoch + 1, config.num_epochs):

            model.train()

            for step, data in enumerate(data_loader):

                real_images = data[0]["data"]
                real_labels = data[0]["label"]

                real_images = real_images.cuda()
                real_labels = real_labels.cuda()

                real_labels = real_labels.squeeze().long()
                fake_labels = real_labels.clone()

                x = torch.arange(config.batch_size).cuda()
                y = torch.arange(config.batch_size).cuda()
                grid_y, grid_x = torch.meshgrid(y, x)
                positions = torch.stack((grid_y.reshape(-1), grid_x.reshape(-1)), dim=-1)
                latents = torch.randn(config.batch_size, config.latent_size).cuda()
                fake_images = generator(torch.cat((positions, latents, fake_labels), dim=-1))
                fake_images = fake_images.reshape(1, int(np.log2(config.batch_size)), -1)

                real_logits = discriminator(real_images.requires_grad_(True))
                fake_logits = discriminator(fake_images.detach().requires_grad_(True))

                real_logits = torch.gather(real_logits, dim=1, index=real_labels.unsqueeze(-1))
                real_logits = real_logits.squeeze(-1)

                fake_logits = torch.gather(fake_logits, dim=1, index=fake_labels.unsqueeze(-1))
                fake_logits = fake_logits.squeeze(-1)

                real_losses = nn.functional.softplus(-real_logits)
                fake_losses = nn.functional.softplus(+fake_logits)
                discriminator_losses = real_losses + fake_losses

                if config.real_gradient_penalty_weight:
                    real_gradients = torch.autograd.grad(
                        outputs=real_logits,
                        inputs=real_images,
                        grad_outputs=torch.ones_like(real_logits),
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    real_gradient_penalties = torch.sum(real_gradients ** 2, dim=(1, 2, 3))
                    discriminator_losses += real_gradient_penalties * config.real_gradient_penalty_weight

                if config.fake_gradient_penalty_weight:
                    fake_gradients = torch.autograd.grad(
                        outputs=fake_logits,
                        inputs=fake_images,
                        grad_outputs=torch.ones_like(fake_logits),
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    fake_gradient_penalties = torch.sum(fake_gradients ** 2, dim=(1, 2, 3))
                    discriminator_losses += fake_gradient_penalties * config.fake_gradient_penalty_weight

                discriminator_loss = torch.mean(discriminator_losses)
                discriminator_optimizer.zero_grad()
                with amp.scale_loss(discriminator_loss, discriminator_optimizer) as scaled_discriminator_loss:
                    scaled_discriminator_loss.backward()
                discriminator_optimizer.step()

                fake_logits = discriminator(fake_images)

                fake_losses = nn.functional.softplus(-fake_logits)
                generator_losses = fake_losses

                if config.mode_seeking_loss_weight:
                    latent_gradients = torch.autograd.grad(
                        outputs=fake_images,
                        inputs=latents,
                        grad_outputs=torch.ones_like(fake_images),
                        retain_graph=True,
                        create_graph=True
                    )[0]
                    mode_seeking_losses = 1.0 / (torch.sum(latent_gradients ** 2, dim=1) + 1.0e-12)
                    generator_losses += mode_seeking_losses * config.mode_seeking_loss_weight

                generator_loss = torch.mean(generator_losses)
                generator_optimizer.zero_grad()
                with amp.scale_loss(generator_loss, generator_optimizer) as scaled_generator_loss:
                    scaled_generator_loss.backward()
                generator_optimizer.step()

                if step % 100 == 0 and global_rank == 0:
                    summary_writer.add_images(
                        tag="generated_images",
                        img_tensor=fake_images
                    )
                    summary_writer.add_scalars(
                        main_tag='training',
                        tag_scalar_dict=dict(loss=loss)
                    )
                    print(f'[training] epoch: {epoch} step: {step} loss: {loss}')

            torch.save(dict(
                generator_state_dict=generator.state_dict(),
                generator_optimizer_state_dict=generator_optimizer.state_dict(),
                discriminator_state_dict=discriminator.state_dict(),
                discriminator_optimizer_state_dict=discriminator_optimizer.state_dict(),
                last_epoch=epoch
            ), f'{config.checkpoint_directory}/epoch_{epoch}')

    summary_writer.close()


if __name__ == '__main__':
    main()
