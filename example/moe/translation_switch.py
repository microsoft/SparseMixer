import logging
import json
import torch

from argparse import Namespace
from dataclasses import dataclass, field
from omegaconf import II

from fairseq import metrics, models
from fairseq.data import encoders
from fairseq.dataclass import ChoiceEnum
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask


logger = logging.getLogger(__name__)


@dataclass
class TranslationSwitchConfig(TranslationConfig):
    num_experts: int = field(
        default=4,
        metadata={"help": "number of experts"},
    )
    router: ChoiceEnum(['SparseMixer', 'SwitchGate']) = field(
        default='SparseMixer',
        metadata={"help": "choice of router"},
    )
    load_balancing: bool = field(
        default=False,
        metadata={"help": "whether to use load balancing"},
    )
    gumbel: bool = field(
        default=False,
        metadata={"help": "use gumbel logits for computing balancing loss"},
    )
    jitter_eps: float = field(
        default=0.1,
        metadata={"help": "jitter eps"},
    )
    load_balancing_alpha: float = field(
        default=0.01,
        metadata={"help": "weight of load balancing loss"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


@register_task("translation_switch", dataclass=TranslationSwitchConfig)
class TranslationSwitchTask(TranslationTask):
    """
    Translation task for Switch Transformer models.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    cfg: TranslationSwitchConfig

    def __init__(self, cfg: TranslationSwitchConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def build_model(self, cfg):
        model = models.build_model(cfg, self)

        if self.cfg.eval_bleu:
            detok_args = json.loads(self.cfg.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )

        return model

    def _get_loss(self, sample, model, criterion):
        assert hasattr(
            criterion, "compute_loss"
        ), "translation_switch task requires the criterion to implement the compute_loss() method"

        encoder_out = model.encoder(
            src_tokens=sample["net_input"]["src_tokens"],
            src_lengths=sample["net_input"]["src_lengths"],
        )
        net_output = model.decoder(
            prev_output_tokens=sample["net_input"]["prev_output_tokens"],
            encoder_out=encoder_out,
            src_lengths=sample["net_input"]["src_lengths"],
        )
        loss, nll_loss = criterion.compute_loss(model, net_output, sample, reduce=True)

        balance_loss = None
        if self.cfg.load_balancing:
            balance_loss = net_output[1]["balance_loss"]
            if encoder_out["balance_loss"] is not None:
                balance_loss = balance_loss + encoder_out["balance_loss"]
            loss = loss + balance_loss * self.cfg.load_balancing_alpha

        if 'load' in net_output[1]:
            load = net_output[1]["load"]
            if encoder_out["load"] is not None:
                load = torch.cat((encoder_out["load"], load), dim=0)
        else:
            load = torch.Tensor([1.])

        sample_size = (
            sample["target"].size(0) if criterion.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "load": (load * 100).long(),
            "balance_loss": balance_loss.data if balance_loss is not None else 0.0,
        }
        return loss, sample_size, logging_output

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        with torch.autograd.profiler.record_function("forward"):
            with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                loss, sample_size, logging_output = self._get_loss(sample, model, criterion)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        metrics.log_scalar(
            "load",
            sum(log["load"] for log in logging_outputs if "load" in log) / torch.cuda.device_count(),
            round=3,
        )

        temp = [log["balance_loss"] for log in logging_outputs if "balance_loss" in log]
        metrics.log_scalar("balance_loss", sum(temp), round=3)
