import argparse
import sys

sys.path.append("../S5fork")

import jax
import jax.numpy as jnp

import dataloaders
from s5.utils.util import str2bool


def run(args):
    # Set randomness...
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(args.jax_seed)
    init_rng, train_rng = jax.random.split(key, num=2)

    quantization_config = ...

    # Create dataset...
    # init_rng, key = jax.random.split(init_rng, num=2)

    if args.experiment == "lorenz":
        from lorenz.model import lorenz_ssm

        model = lorenz_ssm(args, init_rng)
    #     (
    #     trainloader,
    #     valloader,
    #     testloader,
    #     aux_dataloaders,
    #     n_classes,
    #     seq_len,
    #     in_dim,
    #     train_size,
    # ) = dataloaders.lorenz_data_fn("lorenz/data/", seed=args.jax_seed, bsz=args.bsz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, choices=["lorenz"])

    # Quant flags
    parser.add_argument("--aquant", type=int, nargs="?", default=None)
    parser.add_argument("--bquant", type=int, nargs="?", default=None)
    parser.add_argument("--cquant", type=int, nargs="?", default=None)
    parser.add_argument("--dquant", type=int, nargs="?", default=None)
    parser.add_argument("--actquant", type=int, nargs="?", default=None)

    # S5 flags
    parser.add_argument(
        "--blocks", type=int, default=8, help="How many blocks, J, to initialize with"
    )
    parser.add_argument(
        "--conj_sym",
        type=str2bool,
        default=True,
        help="whether to enforce conjugate symmetry",
    )
    parser.add_argument(
        "--n_layers", type=int, default=6, help="Number of layers in the network"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Number of features, i.e. H, " "dimension of layer inputs/outputs",
    )
    parser.add_argument(
        "--ssm_size_base", type=int, default=256, help="SSM Latent size, i.e. P"
    )
    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument("--jax_seed", type=int, default=1337, help="seed randomness")

    args = parser.parse_args()
    print(
        f"""Running experiment: {args.experiment} experiment with quantization bits
    a  : {args.aquant}
    b  : {args.bquant}
    c  : {args.cquant}
    d  : {args.dquant}
    act: {args.actquant}
    """
    )
    run(args)
