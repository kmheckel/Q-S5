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
    parser.add_argument(
        "--ssm_lr_base", type=float, default=1e-3, help="initial ssm learning rate"
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=1,
        help="global learning rate = lr_factor*ssm_lr_base",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay value"
    )
    parser.add_argument(
        "--C_init",
        type=str,
        default="trunc_standard_normal",
        choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
        help="Options for initialization of C: \\"
        "trunc_standard_normal: sample from trunc. std. normal then multiply by V \\ "
        "lecun_normal sample from lecun normal, then multiply by V\\ "
        "complex_normal: sample directly from complex standard normal",
    )
    parser.add_argument(
        "--discretization", type=str, default="zoh", choices=["zoh", "bilinear"]
    )
    parser.add_argument(
        "--clip_eigs",
        type=str2bool,
        default=False,
        help="whether to enforce the left-half plane condition",
    )
    parser.add_argument(
        "--bidirectional",
        type=str2bool,
        default=False,
        help="whether to use bidirectional model",
    )
    parser.add_argument(
        "--dt_min",
        type=float,
        default=0.001,
        help="min value to sample initial timescale params from",
    )
    parser.add_argument(
        "--dt_max",
        type=float,
        default=0.1,
        help="max value to sample initial timescale params from",
    )
    parser.add_argument(
        "--conj_sym",
        type=str2bool,
        default=True,
        help="whether to enforce conjugate symmetry",
    )
    parser.add_argument(
        "--activation_fn",
        default="half_glu1",
        type=str,
        choices=["full_glu", "half_glu1", "half_glu2", "gelu"],
    )
    parser.add_argument(
        "--p_dropout", type=float, default=0.0, help="probability of dropout"
    )
    parser.add_argument(
        "--prenorm",
        type=str2bool,
        default=True,
        help="True: use prenorm, False: use postnorm",
    )
    parser.add_argument(
        "--batchnorm",
        type=str2bool,
        default=True,
        help="True: use batchnorm, False: use layernorm",
    )
    parser.add_argument(
        "--bn_momentum", type=float, default=0.95, help="batchnorm momentum"
    )
    parser.add_argument(
        "--dt_global",
        type=str2bool,
        default=False,
        help="Treat timescale parameter as global parameter or SSM parameter",
    )

    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument("--jax_seed", type=int, default=1337, help="seed randomness")
    parser.add_argument(
        "--opt_config",
        type=str,
        default="standard",
        choices=["standard", "BandCdecay", "BfastandCdecay", "noBCdecay"],
        help="Opt configurations: \\ "
        "standard:       no weight decay on B (ssm lr), weight decay on C (global lr) \\"
        "BandCdecay:     weight decay on B (ssm lr), weight decay on C (global lr) \\"
        "BfastandCdecay: weight decay on B (global lr), weight decay on C (global lr) \\"
        "noBCdecay:      no weight decay on B (ssm lr), no weight decay on C (ssm lr) \\",
    )

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
