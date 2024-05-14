import sys

sys.path.append("../")
sys.path.append("../../S5fork")

from functools import partial

import jax
import jax.numpy as jnp

from s5 import ssm, ssm_init, seq_model, train_helpers

def dynamical_ssm(args, init_rng) -> tuple:
    # Set SSM size and block size
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = ssm_init.make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T
    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * jnp.ones((args.blocks, block_size))).ravel()
    V = jax.scipy.linalg.block_diag(*([V] * args.blocks))
    Vinv = jax.scipy.linalg.block_diag(*([Vc] * args.blocks))

    ssm_init_fn = ssm.init_S5SSM(
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional,
    )

    model_cls = partial(
        seq_model.StackedEncoderModel,
        ssm=ssm_init_fn,
        d_model=args.d_model,
        n_layers=args.n_layers,
        activation=args.activation_fn,
        dropout=args.p_dropout,
        training=True,
        prenorm=args.prenorm,
        batchnorm=args.batchnorm,
        bn_momentum=args.bn_momentum,
    )

    state = train_helpers.create_train_state(
        model_cls,
        init_rng,
        padded=False,
        retrieval=False,
        in_dim=3,
        bsz=args.bsz,
        seq_len=128,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global,
    )
    return model_cls, state
