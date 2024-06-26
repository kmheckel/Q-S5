import sys

sys.path.append("../")
sys.path.append("../../S5fork")

from functools import partial

import jax
import jax.numpy as jnp
import wandb

from s5 import qssm_aqt, ssm_init, qseq_model, train_helpers

def dynamical_ssm(args, seq_len, in_dim, init_rng) -> tuple:
    # Set SSM size and block size
    ssm_size = args.ssm_size_base
    block_size = int(ssm_size / args.blocks)
    wandb.log({"block_size": block_size})

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

    q_config = qssm_aqt.QuantizationConfig(
        a_precision=args.a_bits,
        b_precision=args.b_bits,
        c_precision=args.c_bits,
        d_precision=args.d_bits,
        non_ssm_precision=args.non_ssm_bits,
        ssm_act_precision=args.ssm_act_bits,
        non_ssm_act_precision=args.non_ssm_act_bits,
    )

    ssm_init_fn = qssm_aqt.init_qS5SSM(
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
        q_config=q_config
    )

    model_cls = partial(
        qseq_model.QBatchRegressionModel,
        ssm=ssm_init_fn,
        d_model=args.d_model,
        d_output=in_dim,
        n_layers=args.n_layers,
        q_bits_aw=(q_config.non_ssm_act_precision, q_config.non_ssm_precision),
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
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        batchnorm=args.batchnorm,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        dt_global=args.dt_global,
    )
    return model_cls, state
