import sys
import os
import argparse

import torch
import torch.nn as nn
from torch.nn import functional as F

from egg.core.interaction import LoggingStrategy

from utils.losses import MSE_loss, compute_topsim_zakajd, koleo_loss, pdm  # relies on: MSE_loss, compute_topsim_extended, kl_divergence_loss, z_MSE_loss, wasserstein_loss, load_config, WandbTopSim, EarlyStop, pdm, koleo_loss, linear


def _aux1d(x):
    """Return a 1-D CPU tensor from (tensor|number)."""
    if isinstance(x, torch.Tensor):
        return torch.atleast_1d(x.detach().to('cpu'))
    else:
        return torch.tensor([float(x)], device='cpu')


class linear(torch.nn.Module):
    # a linear activation function based on y=x
    def forward(self, output):return output

class Sender(nn.Module):
    def __init__(self, output_size, activation, vision):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "none":
            self.activation = linear()

        self.vision = vision
        self.fc1 = nn.Linear(output_size // 2, output_size)
        self.fc2 = nn.Linear(output_size, output_size)

    def forward(self, x, aux_input=None):
        x = self.vision.encode(x)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return x


class Receiver(nn.Module):
    def __init__(self, input_size, vision):
        super().__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size // 2)
        self.vision = vision

    def forward(self, channel_input, receiver_input=None, aux_input=None):
        x = F.relu(self.fc1(channel_input))
        x = self.fc2(x)
        return x

    def to_imgs(self, vectors):
        b, c, v = vectors.shape
        x = vectors.reshape(b * c, v)
        x = self.vision.decode(x)
        x = x.reshape(b, c, -1)
        return x


class InteractionGame(nn.Module):
    def __init__(
        self,
        sender,
        receiver,
        loss=MSE_loss,
        lambda_=1,
        interaction_mode=None,
        train_logging_strategy=None,
        test_logging_strategy=None,
        log_top_sim=True,
        epoch=0,
    ):
        super().__init__()
        self.sender = sender
        self.receiver = receiver
        self.loss = loss
        self.lambda_ = lambda_
        self.log_top_sim = log_top_sim
        self.epoch = epoch
        self.interaction_mode = interaction_mode
        self.train_logging_strategy = train_logging_strategy or LoggingStrategy()
        self.test_logging_strategy = test_logging_strategy or LoggingStrategy()

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None, return_messages=False):
        message = self.sender(sender_input, aux_input)
        labels = labels.to(dtype=torch.float, device=sender_input.device)  # kept, even if not used elsewhere

        receiver_output = self.receiver(message)
        receiver_output = self.receiver.agent.to_imgs(receiver_output)

        unrolled_X = sender_input.flatten(start_dim=1).unsqueeze(1).repeat(1, receiver_output.size(1), 1)
        rec_loss = self.loss(receiver_output, unrolled_X)

        total_loss = rec_loss
        n = receiver_output.size(1)
        exponents = torch.arange(n, device=sender_input.device)
        lambda_tensor = torch.pow(self.lambda_, exponents)

        total_loss_log = total_loss.clone().detach()
        total_loss *= lambda_tensor

        if self.interaction_mode == "full":
            total_loss = total_loss[:, -1]

        loss = total_loss.mean()

        # === Logging ===
        logging_strategy = self.train_logging_strategy if self.training else self.test_logging_strategy
        aux_info = {}

        if not self.training:
            with torch.no_grad():
                # Means over batch/time are scalars → make them 1-D
                aux_info["final_symbol_loss"] = _aux1d(total_loss_log[:, -1].mean())
                for i in range(total_loss_log.shape[1]):
                    aux_info[f"loss_position:{i}"] = _aux1d(total_loss_log[:, i].mean())

                discrete_message = message.argmax(dim=-1).float()
                discrete_message[:, :-1] += 1  # Ignore eos marker

                if self.log_top_sim:
                    topsim, message_hamming = compute_topsim_zakajd(
                        labels, discrete_message, "hamming", "hamming"
                    )
                    aux_info["top_sim"] = _aux1d(topsim)
                    aux_info["message_hamming"] = _aux1d(message_hamming)

                # This was 0-dim before; make it 1-D
                aux_info["interaction_loss"] = _aux1d(loss)
                if return_messages:
                    aux_info["messages"] = discrete_message

        interaction = logging_strategy.filtered_interaction(
            sender_input=None,
            receiver_input=None,
            labels=None,
            aux_input=None,
            receiver_output=None,
            message=None,
            message_length=None,
            aux=aux_info,
        )

        return loss, interaction

    def predict(self, sender_input):
        message = self.sender(sender_input)
        receiver_output = self.receiver(message)
        return sender_input, message, receiver_output

    def interpolate(self, sender_input_a, sender_input_b):
        message_a = self.sender(sender_input_a)
        message_b = self.sender(sender_input_b)

        outputs = [self.receiver(message_a)]
        for i in range(1, message_a.size(1) + 1):
            left = message_a[:, :-i, :] if i < message_a.size(1) else torch.empty_like(message_b[:, :0, :])
            message = torch.cat((left, message_b[:, -i:, :]), dim=1)
            receiver_output = self.receiver(message)
            outputs.append(receiver_output)

        return outputs


class ImitationGame(nn.Module):
    def __init__(
        self,
        teacher,
        student,
        receiver,
        imitation_mode,
        epy_mode="epy",
        loss=MSE_loss,
        beta=1,
        train_logging_strategy=None,
        test_logging_strategy=None,
        device="cuda",
    ):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.loss = loss
        self.receiver = receiver
        self.imitation_mode = imitation_mode
        self.beta = beta
        self.sim_loss = pdm if epy_mode == "epy" else koleo_loss
        self.train_logging_strategy = train_logging_strategy or LoggingStrategy()
        self.test_logging_strategy = test_logging_strategy or LoggingStrategy()
        self.device = device

    def forward(self, sender_input, labels, receiver_input=None, aux_input=None):
        distributions = sender_input

        with torch.no_grad():
            teacher_message = self.teacher(distributions, aux_input)
            if self.imitation_mode != "msg":
                if self.imitation_mode == "last":
                    t_out = self.receiver(teacher_message, receiver_input, aux_input)[:, -1, :][:, None, :]
                    teacher_receiver_output = self.receiver.agent.to_imgs(t_out)
                else:
                    t_out = self.receiver(teacher_message, receiver_input, aux_input)
                    teacher_receiver_output = self.receiver.agent.to_imgs(t_out)

        student_message = self.student(distributions, aux_input)
        if self.imitation_mode != "msg":
            if self.imitation_mode == "last":
                s_out = self.receiver(student_message, receiver_input, aux_input)[:, -1, :][:, None, :]
                student_receiver_output = self.receiver.agent.to_imgs(s_out)
            else:
                s_out = self.receiver(student_message, receiver_input, aux_input)
                student_receiver_output = self.receiver.agent.to_imgs(s_out)

        if self.imitation_mode == "msg":
            rec_loss = F.nll_loss(
                torch.log(student_message.transpose(-1, 1)),
                teacher_message.argmax(dim=-1).to(dtype=torch.long, device="cuda"),
            )
        else:
            rec_loss = self.loss(student_receiver_output, teacher_receiver_output).mean()

        sim_loss = torch.zeros(1, device="cuda")
        if self.beta > 0:
            sim_loss = self.sim_loss(student_message, self.student.straight_through).mean()

        loss = rec_loss + self.beta * sim_loss
        logging_strategy = self.train_logging_strategy if self.training else self.test_logging_strategy

        interaction = logging_strategy.filtered_interaction(
            sender_input=None,
            receiver_input=None,
            labels=None,
            aux_input=None,
            receiver_output=None,
            message=None,
            message_length=None,
            aux={
                "imitation_loss": torch.tensor([loss.detach()]),
                "sim_loss": torch.tensor([sim_loss.detach()]),
                "rec_loss": torch.tensor([rec_loss.detach()]),
            },
        )
        return loss, interaction
