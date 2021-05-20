import argparse


def load_config():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=160)
    parser.add_argument('--print_intervals', type=int, default=100)
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--checkpoints', type=str, default=None, help='model checkpoints path')
    parser.add_argument('--device_num', type=int, default=1)
    parser.add_argument('--gradient_clip', type=float, default=1.)
    parser.add_argument('--patch_size', type=int, default=32)
    parser.add_argument('--N', type=int, default=8)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--dims', type=int, default=512)

    return parser.parse_args()
