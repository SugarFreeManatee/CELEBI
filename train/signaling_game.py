import sys
import os
import argparse

import torch
import egg.core as core

from sklearn.model_selection import train_test_split

from disent.frameworks.vae import BetaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, EncoderConv64

import utils.compositional_data as comp
from utils.losses import MSE_loss, kl_divergence_loss, z_MSE_loss, wasserstein_loss  
from utils.compositional_data import load_config
from utils.callbacks import EarlyStop, WandbTopSim

from models import Sender, Receiver, InteractionGame, ImitationGame

def get_params(params):
    parser = argparse.ArgumentParser(description="Training script for signaling game with CNN.")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--emb_size", type=int, default=64)
    parser.add_argument("--v_size", type=int, default=10)
    parser.add_argument("--b_size_im", type=int, default=128)
    parser.add_argument("--b_size_int", type=int, default=128)
    parser.add_argument("--n_epoch", type=int, default=10)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=20)
    parser.add_argument("--sender_temp", type=float, default=1.0)
    parser.add_argument("--student_temp", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--cell", type=str, default="gru")
    parser.add_argument("--mode", type=str, default="ts")
    parser.add_argument("--project", type=str, default="cvpr_final")
    parser.add_argument("--im_epochs", type=int, default=1)
    parser.add_argument("--int_epochs", type=int, default=1)
    parser.add_argument("--lambda_", type=float, default=1.5)
    parser.add_argument("--loss", type=str, default="kldiv")
    parser.add_argument("--config", type=str, default="kldiv")
    parser.add_argument("--sweep", type=str, default="")
    args = core.init(arg_parser=parser, params=params)
    return args


def main(params):
    opts = get_params(params)

    config = load_config(opts.config)
    hidden_size = opts.hidden_size
    emb_size = opts.emb_size
    vocab_size = opts.v_size
    batch_size_im = opts.b_size_im
    batch_size_int = opts.b_size_int
    n_epoch = opts.n_epoch
    beta = opts.beta
    max_length = opts.max_length - 1
    torch.manual_seed(opts.random_seed)
    lr = config["lr"]
    activation = opts.activation
    mode = opts.mode

    if ("epy" not in mode) and ("koleo" not in mode):
        beta = 0

    imitation_mode = ""
    interaction_mode = None
    if "last" in mode:
        imitation_mode = "last"
    if "msg" in mode:
        imitation_mode = "msg"
    if "full" in mode:
        interaction_mode = "full"

    epy_mode = "epy" if "koleo" not in mode else "koleo"

    if opts.loss == "kldiv":
        loss = kl_divergence_loss
    elif opts.loss == "zMSE":
        loss = z_MSE_loss
    elif opts.loss == "wasserstein":
        loss = wasserstein_loss
    else:
        loss = MSE_loss

    opts.validation_freq = 10
    int_epochs = opts.int_epochs
    im_epochs = opts.im_epochs
    shapes_dir = config["shapes_dir"]

    train, test = comp._load_shapes3d(shapes_dir) if "shapes3d" in shapes_dir else comp._load_mpi3d(shapes_dir)
    train, _ = train_test_split(train, test_size=0.1, random_state=42)

    kwargs = {"num_workers": config["num_workers"], "pin_memory": True} if torch.cuda.is_available() else {}

    train_loader_int = torch.utils.data.DataLoader(train, batch_size=batch_size_int, shuffle=True, **kwargs)
    train_loader_im = torch.utils.data.DataLoader(train, batch_size=batch_size_im, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size_im, shuffle=False, **kwargs)

    vision = BetaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=train[0][0].shape, z_size=128, z_multiplier=2),
            decoder=DecoderConv64(x_shape=train[0][0].shape, z_size=128),
        )
    )

    vision_receiver = BetaVae(
        model=AutoEncoder(
            encoder=EncoderConv64(x_shape=train[0][0].shape, z_size=128, z_multiplier=2),
            decoder=DecoderConv64(x_shape=train[0][0].shape, z_size=128),
        )
    )

    vision.load_state_dict(torch.load(config["pretrain"]))
    vision_receiver.load_state_dict(torch.load(config["pretrain"]))

    for param in vision._model.parameters():
        param.requires_grad = False
    for param in vision_receiver._model.parameters():
        param.requires_grad = False

    if "end2end" in mode:
        for param in vision_receiver._model._decoder.parameters():
            param.requires_grad = True

    sender1 = Sender(hidden_size, activation=activation, vision=vision).cuda()
    sender2 = Sender(hidden_size, activation=activation, vision=vision).cuda()
    receiver = Receiver(hidden_size, vision=vision_receiver).cuda()

    sender1 = core.RnnSenderGS(
        sender1,
        vocab_size,
        emb_size,
        hidden_size,
        cell=opts.cell,
        max_len=max_length,
        temperature=opts.sender_temp,
        trainable_temperature=False,
        straight_through=True,
    ).cuda()
    receiver = core.RnnReceiverGS(receiver, vocab_size, emb_size, hidden_size, cell=opts.cell).cuda()
    sender2 = core.RnnSenderGS(
        sender2,
        vocab_size,
        emb_size,
        hidden_size,
        cell=opts.cell,
        max_len=max_length,
        temperature=opts.student_temp,
        trainable_temperature=False,
        straight_through=False,
    ).cuda()

    start_epoch = 0

    int_game = InteractionGame(
        sender1,
        receiver,
        lambda_=opts.lambda_,
        interaction_mode=interaction_mode,
        log_top_sim=True,  # logging disabled by default
        epoch=start_epoch,
        loss=loss
    ).cuda()

    im_game = ImitationGame(
        sender1, sender2, receiver, beta=beta, imitation_mode=imitation_mode, epy_mode=epy_mode, loss=loss
    ).cuda()

    optimizer1 = torch.optim.Adam(int_game.parameters(), lr=lr)
    optimizer2 = torch.optim.Adam(sender2.parameters(), lr=lr)

    logger = WandbTopSim(opts=opts, project=opts.project, epoch=start_epoch)
    stopper = EarlyStop(patience=5, min_delta=1e-3)

    trainer1 = core.Trainer(
        game=int_game,
        optimizer=optimizer1,
        train_data=train_loader_int,
        validation_data=test_loader,
        callbacks=[logger, stopper],
    )
    trainer1.validation_freq = 1

    trainer2 = core.Trainer(
        game=im_game,
        optimizer=optimizer2,
        train_data=train_loader_im,
        validation_data=test_loader,
        callbacks=[logger],
    )

    redundant_hyp = (("ts" not in mode or beta == 0) and opts.beta > 1) or (("full" in mode) and (opts.lambda_ != 1)) or (
        "ts" not in mode and "msg" in mode
    )

    save_dir = os.path.join(
        config["checkpoint_dir"],
        opts.sweep,
        f"sender;mode:cnn_{mode};beta:{beta};lambda:{opts.lambda_};dataset:{shapes_dir.split('/')[-1]}",
    )
    os.makedirs(save_dir, exist_ok=True)

    best = (sender1.state_dict(), receiver.state_dict(), 0)
    for epoch in range(start_epoch, n_epoch):
        if stopper.stop or redundant_hyp:
            print("Early stop")
            break

        if ("ts" in mode) and (epoch > 0):
            if epoch % int_epochs == 0:
                if "end2end" in mode:
                    for param in receiver.agent.vision._model._decoder.parameters():
                        param.requires_grad = False
                trainer2.train(im_epochs)
                sender1.agent.load_state_dict(sender2.agent.state_dict())
                sender2.reset_parameters()
                sender2.agent.fc1.reset_parameters()
                sender2.agent.fc2.reset_parameters()
                if "end2end" in mode:
                    for param in receiver.agent.vision._model._decoder.parameters():
                        param.requires_grad = True

        trainer1.train(1)

        if (epoch % 100 == 0) and (epoch > 0):
            torch.save(sender1.state_dict(), os.path.join(save_dir, f"sender;seed:{opts.random_seed};epoch:{epoch}"))
            torch.save(receiver.state_dict(), os.path.join(save_dir, f"receiver;seed:{opts.random_seed};epoch:{epoch}"))

        if stopper.save:
            best = (sender1.state_dict(), receiver.state_dict(), epoch)

        int_game.epoch += 1

    if not redundant_hyp:
        torch.save(sender1.state_dict(), os.path.join(save_dir, f"sender;seed:{opts.random_seed};epoch:{epoch}"))
        torch.save(receiver.state_dict(), os.path.join(save_dir, f"receiver;seed:{opts.random_seed};epoch:{epoch}"))
        torch.save(best[0], os.path.join(save_dir, f"sender;seed:{opts.random_seed};epoch:{best[2]}_best"))
        torch.save(best[1], os.path.join(save_dir, f"receiver;seed:{opts.random_seed};epoch:{best[2]}_best"))

    core.close()


if __name__ == "__main__":
    main(sys.argv[1:])
