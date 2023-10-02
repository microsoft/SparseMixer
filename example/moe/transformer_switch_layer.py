import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Optional

from sparsemixer import get_router

from fairseq.modules.quant_noise import quant_noise
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer


def use_switch(layer_idx):
    return layer_idx % 2 == 0


class SwitchTransformerEncoderLayer(TransformerEncoderLayer):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, layer_idx=-1):
        self.num_experts = args.num_experts
        self.load_balancing = args.load_balancing
        self.use_switch = use_switch(layer_idx)
        super().__init__(args)
        self.gating_network = nn.Linear(args.encoder_embed_dim, args.num_experts)
        if self.use_switch:
            self.router = get_router(args.router)(args.num_experts, args.encoder_embed_dim, args.load_balancing, args.jitter_eps)
        else:
            self.router = None

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_switch:
            return nn.ModuleList(
                [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
                 for _ in range(self.num_experts)]
            )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_switch:
            return nn.ModuleList(
                [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
                 for _ in range(self.num_experts)]
            )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        num_tokens = None
        balance_loss = 0.0
        if self.use_switch:
            seq_len, bsz, dim = x.shape
            x = x.view(-1, dim)
            logits = self.gating_network(x)
            sample, multiplier, balance_loss = self.router(logits)
            
            order = sample.argsort(0).squeeze(-1)
            num_tokens = F.one_hot(sample.squeeze(), self.num_experts).gt(0).sum(0)
            x = x[order]  # reorder according to expert number
            x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts
            
            def forward_fc(input_x, expert_idx):
                if input_x.numel() > 0:
                    input_x = self.activation_fn(self.fc1[expert_idx](input_x))
                    input_x = self.activation_dropout_module(input_x)
                    input_x = self.fc2[expert_idx](input_x)
                return input_x
            x = torch.vstack(
                [forward_fc(x[i], i) for i in range(self.num_experts)]
            )
            x = x[order.argsort()] * multiplier
                
            x = x.view(seq_len, bsz, dim)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, num_tokens, balance_loss


class SwitchTransformerDecoderLayer(TransformerDecoderLayer):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, layer_idx=-1,
    ):
        self.num_experts = args.num_experts
        self.load_balancing = args.load_balancing
        self.use_switch = use_switch(layer_idx)
        super().__init__(args, no_encoder_attn, add_bias_kv, add_zero_attn)
        self.gating_network = nn.Linear(args.decoder_embed_dim, args.num_experts)
        if self.use_switch:
            self.router = get_router(args.router)(args.num_experts, args.decoder_embed_dim, args.load_balancing, args.jitter_eps)
        else:
            self.router = None

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_switch:
            return nn.ModuleList(
                [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
                 for _ in range(self.num_experts)]
            )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        if self.use_switch:
            return nn.ModuleList(
                [quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)
                 for _ in range(self.num_experts)]
            )
        else:
            return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        num_tokens = None
        balance_loss = 0.0
        if self.use_switch:
            seq_len, bsz, dim = x.shape
            x = x.view(-1, dim)
            logits = self.gating_network(x)
            sample, multiplier, balance_loss = self.router(logits)
            
            order = sample.argsort(0).squeeze(-1)
            num_tokens = F.one_hot(sample.squeeze(), self.num_experts).gt(0).sum(0)
            x = x[order]  # reorder according to expert number
            x = x.split(num_tokens.tolist(), dim=0)  # a list of length self.num_experts
            
            def forward_fc(input_x, expert_idx):
                if input_x.numel() > 0:
                    input_x = self.activation_fn(self.fc1[expert_idx](input_x))
                    input_x = self.activation_dropout_module(input_x)
                    input_x = self.fc2[expert_idx](input_x)
                return input_x
            x = torch.vstack(
                [forward_fc(x[i], i) for i in range(self.num_experts)]
            )
            x = x[order.argsort()] * multiplier
            
            x = x.view(seq_len, bsz, dim)
        else:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            x = self.fc2(x)

        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, num_tokens, balance_loss
