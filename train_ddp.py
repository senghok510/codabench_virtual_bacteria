"""
Multi-node PyTorch DDP training: a small CNN on MNIST (handwritten digits 0-9).

Launch on EACH machine with torchrun (see HOW TO LAUNCH at the bottom).
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

# MNIST is stored on each machine's local disk (/Data is not shared between machines).
DATA_DIR = f"/Data/{os.environ.get('USER', 'user')}/mnist"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["small", "large"], default="small",
                        help="large = much more compute per step, to show the parallel speedup")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128, help="per GPU")
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def setup():
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return device, local_rank


def get_datasets(local_rank):
    # Standard MNIST normalisation (mean and std of the training pixels).
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    # One process per machine downloads; the others wait, so they don't
    # write the same files at the same time.
    if local_rank == 0:
        datasets.MNIST(DATA_DIR, train=True, download=True)
        datasets.MNIST(DATA_DIR, train=False, download=True)
    dist.barrier()
    train_set = datasets.MNIST(DATA_DIR, train=True, transform=transform)
    test_set = datasets.MNIST(DATA_DIR, train=False, transform=transform)
    return train_set, test_set


class Net(nn.Module):
    """Two conv layers + two linear layers, about 99% test accuracy."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28 -> 14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def conv_block(c_in, c_out):
    return [nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


class LargeNet(nn.Module):
    """Wide CNN: lots of computation per image but only ~1.1M parameters.

    Machines exchange one gradient per parameter each step, so few parameters
    and heavy computation is the case where adding machines pays off.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            *conv_block(1, 64), *conv_block(64, 128), *conv_block(128, 128),  # 28x28
            nn.MaxPool2d(2),
            *conv_block(128, 256), *conv_block(256, 256),                      # 14x14
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, device):
    """Test accuracy over the whole test set, combined across all machines."""
    model.eval()
    stats = torch.zeros(2, device=device)  # [correct, count]
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        stats[0] += (model(xb).argmax(dim=1) == yb).sum()
        stats[1] += len(yb)
    dist.all_reduce(stats)
    model.train()
    return (stats[0] / stats[1]).item()


def main():
    args = parse_args()
    device, local_rank = setup()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == 0:
        print(f"World size: {world_size} processes")
    print(f"[rank {rank}] host={os.uname().nodename} device={device}", flush=True)

    train_set, test_set = get_datasets(local_rank)
    # Each process trains on a different 1/world_size share of the data.
    train_sampler = DistributedSampler(train_set, shuffle=True)
    test_sampler = DistributedSampler(test_set, shuffle=False)
    loader_kwargs = dict(batch_size=args.batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, **loader_kwargs)
    test_loader = DataLoader(test_set, sampler=test_sampler, **loader_kwargs)
    if rank == 0:
        print(f"Train: {len(train_set)} images, {len(train_loader)} steps/epoch per process. "
              f"Test: {len(test_set)} images.", flush=True)

    model = (LargeNet() if args.model == "large" else Net()).to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model: {args.model}, {n_params / 1e6:.2f}M parameters "
              f"({n_params * 4 / 1e6:.1f} MB of gradients sent per step)", flush=True)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Start the clock only once every machine is ready, so waiting for the
    # other machine to join is not counted as training time.
    dist.barrier()
    train_start = time.perf_counter()

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        train_sampler.set_epoch(epoch)  # reshuffle differently each epoch
        stats = torch.zeros(3, device=device)  # [loss sum, correct, count]
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()          # gradients are averaged across all GPUs here
            optimizer.step()

            stats[0] += loss.detach() * len(yb)
            stats[1] += (logits.argmax(dim=1) == yb).sum()
            stats[2] += len(yb)
            if rank == 0 and step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)}: "
                      f"loss={loss.item():.4f}", flush=True)

        # Combine the numbers from all processes for logging.
        dist.all_reduce(stats)
        train_loss = (stats[0] / stats[2]).item()
        train_acc = (stats[1] / stats[2]).item()
        epoch_time = time.perf_counter() - epoch_start
        test_acc = evaluate(model, test_loader, device)
        if rank == 0:
            # Images per second over all machines: the number that should grow
            # when you add machines.
            throughput = stats[2].item() / epoch_time
            print(f"epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.2%} "
                  f"test_acc={test_acc:.2%} time={epoch_time:.2f}s "
                  f"throughput={throughput:,.0f} img/s", flush=True)

    # Wait for the slowest machine before stopping the clock.
    dist.barrier()
    train_time = time.perf_counter() - train_start
    if rank == 0:
        print(f"Total training time: {train_time:.2f}s on {world_size} processes "
              f"(final test_acc={test_acc:.2%})", flush=True)
        torch.save(model.module.state_dict(), "mnist_cnn.pt")  # save once, from rank 0
        print("Saved mnist_cnn.pt")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

# ---------------------------------------------------------------------------
# HOW TO LAUNCH (2 machines: requin = node 0 / master, saumon = node 1)
#
# On requin:
#   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=0 \
#            --master_addr=requin.polytechnique.fr --master_port=29500 train_ddp.py
#
# On saumon:
#   torchrun --nnodes=2 --nproc_per_node=1 --node_rank=1 \
#            --master_addr=requin.polytechnique.fr --master_port=29500 train_ddp.py
#
# Options (same on both machines): --epochs 10 --batch-size 128 --lr 1e-3
# One machine only, to compare times:
#   torchrun --nnodes=1 --nproc_per_node=1 train_ddp.py
#
# Set --nproc_per_node to the number of GPUs on each machine (check with nvidia-smi).
# ---------------------------------------------------------------------------
