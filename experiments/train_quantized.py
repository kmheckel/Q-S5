import sys

sys.path.append("../S5fork")

import jax

from s5.train_helpers import (
    create_train_state,
    reduce_lr_on_plateau,
    linear_warmup,
    cosine_annealing,
    constant_lr,
    train_epoch,
    validate,
)


def train(
    args,
    model_cls,
    trainloader,
    valloader,
    testloader,
    seq_len,
    in_dim,
    state,
    train_rng,
    loss_fn
):
    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    ssm_lr = args.ssm_lr_base
    lr = args.lr_factor * ssm_lr

    # Training Loop over epochs
    best_loss, best_acc, best_epoch = (
        100000000,
        -100000000.0,
        0,
    )  # This best loss is val_loss
    count, best_val_loss = 0, 100000000  # This line is for early stopping purposes
    lr_count, opt_acc = 0, -100000000.0  # This line is for learning rate decay
    step = 0  # for per step learning rate decay
    steps_per_epoch = int(len(trainloader) / args.bsz)
    for epoch in range(args.epochs):
        print(f"[*] Starting Training Epoch {epoch + 1}...")

        if epoch < args.warmup_end:
            print("using linear warmup for epoch {}".format(epoch + 1))
            decay_function = linear_warmup
            end_step = steps_per_epoch * args.warmup_end

        elif args.cosine_anneal:
            print("using cosine annealing for epoch {}".format(epoch + 1))
            decay_function = cosine_annealing
            # for per step learning rate decay
            end_step = steps_per_epoch * args.epochs - (
                steps_per_epoch * args.warmup_end
            )
        else:
            print("using constant lr for epoch {}".format(epoch + 1))
            decay_function = constant_lr
            end_step = None

        # TODO: Switch to letting Optax handle this.
        #  Passing this around to manually handle per step learning rate decay.
        lr_params = (
            decay_function,
            ssm_lr,
            lr,
            step,
            end_step,
            args.opt_config,
            args.lr_min,
        )

        train_rng, skey = jax.random.split(train_rng)
        state, train_loss, step = train_epoch(
            state,
            skey,
            model_cls,
            trainloader,
            seq_len,
            in_dim,
            batchnorm=args.batchnorm,
            lr_params=lr_params,
            loss_fn=loss_fn
        )

        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(
                state, model_cls, valloader, seq_len, in_dim, args.batchnorm
            )

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(
                state, model_cls, testloader, seq_len, in_dim, args.batchnorm
            )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f} --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

        else:
            # # else use test set as validation set (e.g. IMDB)
            # print(f"[*] Running Epoch {epoch + 1} Test...")
            # val_loss, val_acc = validate(
            #     state, model_cls, testloader, seq_len, in_dim, args.batchnorm
            # )

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f}  --Test Loss: {val_loss:.5f} --"
                f" Test Accuracy: {val_acc:.4f}"
            )

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            # Increment counters etc.
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # For learning rate decay purposes:
        input = lr, ssm_lr, lr_count, val_acc, opt_acc
        lr, ssm_lr, lr_count, opt_acc = reduce_lr_on_plateau(
            input,
            factor=args.reduce_factor,
            patience=args.lr_patience,
            lr_min=args.lr_min,
        )

        # Print best accuracy & loss so far...
        print(
            f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}\n"
            f"\tBest Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
        )
