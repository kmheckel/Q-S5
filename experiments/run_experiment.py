import argparse
import sys
import pathlib

sys.path.append("../S5fork")
# sys.path.append("../../S5")

import jax
import jax.numpy as jnp
import optax
import wandb

import dataloaders
from s5.utils.util import str2bool
from train_quantized import train


def run(args):
    # Setup logging
    wandb.init(mode='offline', project=args.experiment)

    # Set randomness...
    print("[*] Setting Randomness...")
    key = jax.random.PRNGKey(args.jax_seed)
    init_rng, train_rng = jax.random.split(key, num=2)

    if args.experiment == "lorenz":
        from dynamical.model import dynamical_ssm

        data_path = pathlib.Path(__file__).parent / "dynamical" / "data" / "Lorenz"
        (
            trainloader,
            valloader,
            testloader,
            aux_dataloaders,
            n_classes,
            seq_len,
            in_dim,
            train_size,
        ) = dataloaders.dysts_data_fn(
            str(data_path), timesteps=args.timesteps, seed=key, bsz=args.bsz
        )

        args.d_model = in_dim
        model, state = dynamical_ssm(args, seq_len, in_dim, init_rng)
        loss_fn = optax.squared_error

    elif args.experiment == "mackey_glass":
        from dynamical.model import dynamical_ssm

        # modify this to be a loop over the various tau values and their directories.
        data_path = pathlib.Path(__file__).parent / "dynamical" / "data" / "MackeyGlass"
        (
            trainloader,
            valloader,
            testloader,
            aux_dataloaders,
            n_classes,
            seq_len,
            in_dim,
            train_size,
        ) = dataloaders.dysts_data_fn(
            str(data_path), timesteps=args.timesteps, seed=key, bsz=args.bsz
        )

        args.d_model = in_dim
        model, state = dynamical_ssm(args, seq_len, in_dim, init_rng)
        loss_fn = optax.squared_error

    else:
        raise NotImplementedError()

    train(
        args,
        model_cls=model,
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        seq_len=seq_len,
        in_dim=in_dim,
        state=state,
        train_rng=train_rng,
        loss_fn=loss_fn,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, choices=["lorenz", "dynamical_all"])

    # Quant flags
    parser.add_argument("--a_bits", type=int, nargs="?", default=None)
    parser.add_argument("--b_bits", type=int, nargs="?", default=None)
    parser.add_argument("--c_bits", type=int, nargs="?", default=None)
    parser.add_argument("--d_bits", type=int, nargs="?", default=None)
    parser.add_argument("--ssm_act_bits", type=int, nargs="?", default=None)
    parser.add_argument("--non_ssm_bits", type=int, nargs="?", default=None)
    parser.add_argument("--non_ssm_act_bits", type=int, nargs="?", default=None)

    # Dataset flags
    parser.add_argument(
        "--timesteps",
        type=int,
        default=512,
        help="number of timesteps for dynamical systems",
    )

    # S5 flags
    parser.add_argument(
        "--blocks", type=int, default=2, help="How many blocks, J, to initialize with"
    )
    parser.add_argument(
        "--n_layers", type=int, default=2, help="Number of layers in the network"
    )
    parser.add_argument("--epochs", type=int, default=1000, help="max number of epochs")
    parser.add_argument(
        "--warmup_end", type=int, default=1, help="epoch to end linear warmup"
    )
    parser.add_argument(
        "--ssm_size_base", type=int, default=8, help="SSM Latent size, i.e. P"
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
    parser.add_argument("--lr_min", type=float, default=0, help="minimum learning rate")
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1000000,
        help="patience before decaying learning rate for lr_decay_on_val_plateau",
    )
    parser.add_argument(
        "--reduce_factor",
        type=float,
        default=1.0,
        help="factor to decay learning rate for lr_decay_on_val_plateau",
    )
    parser.add_argument("--cosine_anneal", type=str2bool, default=True,
						help="whether to use cosine annealing schedule")
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
        default="gelu",
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
    a   : {args.a_bits}
    b   : {args.b_bits}
    c   : {args.c_bits}
    d   : {args.d_bits}
    act : {args.ssm_act_bits}
    misc: {args.non_ssm_bits}
    nact: {args.non_ssm_act_bits}
    """
    )
    run(args)
