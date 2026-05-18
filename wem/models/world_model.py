from __future__ import annotations

import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union

from transformers import Qwen3VLForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model


class QueryEmbeddingModule(nn.Module):
    def __init__(
        self,
        num_queries: int = 256,
        hidden_size: int = 3584,
        init_std: float = 0.02,
        role: Optional[str] = None,
    ):
        super().__init__()
        self.num_queries = int(num_queries)
        self.hidden_size = int(hidden_size)
        self.role = role
        self.query_embeddings = nn.Parameter(torch.empty(self.num_queries, self.hidden_size))
        self.role_bias = nn.Parameter(torch.empty(1, self.hidden_size)) if role is not None else None
        self._init_weights(init_std)

    def _init_weights(self, std: float):
        nn.init.trunc_normal_(self.query_embeddings, std=std)
        if self.role_bias is not None:
            nn.init.trunc_normal_(self.role_bias, std=std * 0.5)

    def forward(self, batch_size: int = 1) -> torch.Tensor:
        q = self.query_embeddings if self.role_bias is None else self.query_embeddings + self.role_bias
        return q.unsqueeze(0).expand(batch_size, -1, -1)


class Qwen3WorldModel(nn.Module):
    VIDEO_TOKEN_ID = 151654

    def __init__(
        self,
        model_name: str = os.environ.get("QWEN_CKPT_DIR", "checkpoints/Qwen3-VL-2B-Instruct"),
        num_query_tokens: int = 256,
        num_world_query_tokens: Optional[int] = None,
        num_ego_query_tokens: Optional[int] = None,
        ego_recent_turns: int = 1,
        freeze_backbone: bool = True,
        device: str = "cuda",
        use_device_map: bool = True,
        device_map: Optional[Union[str, Dict[str, object]]] = None,
        cache_dir: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        use_lora: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[List[str]] = None,
    ):
        super().__init__()
        self.num_world_query_tokens = int(num_world_query_tokens or num_query_tokens)
        self.num_ego_query_tokens = int(num_ego_query_tokens or num_query_tokens)
        self.num_query_tokens = self.num_world_query_tokens + self.num_ego_query_tokens
        self.ego_recent_turns = int(max(1, ego_recent_turns))
        self.device = device
        self.dtype = dtype
        self.use_lora = use_lora

        if use_device_map:
            device_map = device_map if device_map is not None else self.device
        else:
            device_map = None

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=self.dtype,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        self.hidden_size = self.backbone.config.text_config.hidden_size

        if use_lora:
            if lora_target_modules is None:
                lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
            )
            self.backbone.model.language_model = get_peft_model(
                self.backbone.model.language_model, lora_config
            )
            self.backbone.model.language_model = self.backbone.model.language_model.to(dtype=dtype)
            self.backbone.model.language_model.print_trainable_parameters()

        if freeze_backbone:
            self._freeze_backbone()

        self.world_query_embeddings = QueryEmbeddingModule(
            num_queries=self.num_world_query_tokens,
            hidden_size=self.hidden_size,
            role='world',
        ).to(device=self.device, dtype=self.dtype)

        self.ego_query_embeddings = QueryEmbeddingModule(
            num_queries=self.num_ego_query_tokens,
            hidden_size=self.hidden_size,
            role='ego',
        ).to(device=self.device, dtype=self.dtype)

    def _freeze_backbone(self):
        for name, param in self.backbone.named_parameters():
            if self.use_lora and "lora_" in name:
                continue
            param.requires_grad = False
        print("Backbone non-LoRA parameters frozen.")

    def _build_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_turn_ids: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """
        Build 4D additive attention mask with the following visibility rules:
        1) ids/visual tokens (base tokens) are causal among themselves.
        2) World queries see valid base tokens from turns strictly before
           the current turn. They do NOT see the current-turn text (n-th text)
           or the user/assistant structural tokens of the current turn, and they
           do NOT see ego queries. World queries see each other.
        3) Ego queries see base tokens from the last K+1 turns, i.e.
           token_turn_ids in [current_turn - K, current_turn]. Given the
           dataset layout where turn t contains "t-th video + (t+1)-th text",
           with K=1 this covers exactly "(n-K) video + (n-K+1) text + ... +
           n text" (= "(n-1) video + n text"). They see each other but not
           world queries.
        """
        B, L = input_ids.shape
        device = input_ids.device
        total_len = L + self.num_query_tokens
        world_start = L
        world_end = world_start + self.num_world_query_tokens
        ego_start = world_end
        ego_end = ego_start + self.num_ego_query_tokens

        valid = attention_mask.bool()  # (B, L)

        # Base tokens: causal + both positions valid — fully vectorized.
        causal = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
        base_allow = causal[None] & valid[:, :, None] & valid[:, None, :]  # (B, L, L)

        # Current turn per batch element: max turn id among valid positions.
        # Mask padding positions with -1 so they don't inflate the max.
        current_turn = token_turn_ids.masked_fill(~valid, -1).amax(dim=1).clamp(min=0)  # (B,)

        # World queries: valid tokens from turns strictly before current_turn.
        world_visible = valid & (token_turn_ids < current_turn[:, None])  # (B, L)

        # Ego queries: valid tokens in [current_turn - K, current_turn].
        min_turn = current_turn - self.ego_recent_turns  # (B,)
        ego_visible = (
            valid
            & (token_turn_ids >= min_turn[:, None])
            & (token_turn_ids <= current_turn[:, None])
        )  # (B, L)

        allow = torch.zeros(B, total_len, total_len, device=device, dtype=torch.bool)
        allow[:, :L, :L] = base_allow
        allow[:, world_start:world_end, :L] = world_visible[:, None, :]
        allow[:, world_start:world_end, world_start:world_end] = True
        allow[:, ego_start:ego_end, :L] = ego_visible[:, None, :]
        allow[:, ego_start:ego_end, ego_start:ego_end] = True

        mask_dtype = dtype if dtype is not None else torch.float32
        neg_inf = torch.finfo(mask_dtype).min
        attention_mask_4d = torch.full(
            (B, 1, total_len, total_len), neg_inf, device=device, dtype=mask_dtype
        )
        attention_mask_4d.masked_fill_(allow[:, None, :, :], 0.0)
        return attention_mask_4d

    def build_input_sequence(
        self,
        ids: torch.Tensor,
        mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        token_turn_ids: torch.Tensor,
    ):
        B = ids.shape[0]
        embed_fn = (
            self.backbone.model.language_model.embed_tokens
            if not self.use_lora
            else self.backbone.model.language_model.model.embed_tokens
        )
        text_embeds = embed_fn(ids)

        for b in range(B):
            video_pos = ids[b] == self.VIDEO_TOKEN_ID
            n = int(video_pos.sum())
            if n > 0:
                if visual_embeds.shape[1] < n:
                    raise ValueError(
                        f"Not enough visual embeddings for batch {b}: "
                        f"{visual_embeds.shape[1]} provided, {n} needed."
                    )
                text_embeds[b, video_pos] = visual_embeds[b, :n].to(text_embeds.dtype)

        world_query = self.world_query_embeddings(B).to(device=text_embeds.device, dtype=text_embeds.dtype)
        ego_query = self.ego_query_embeddings(B).to(device=text_embeds.device, dtype=text_embeds.dtype)
        hidden_states = torch.cat([text_embeds, world_query, ego_query], dim=1)

        attn_mask = self._build_mask(
            input_ids=ids,
            attention_mask=mask,
            token_turn_ids=token_turn_ids,
            dtype=text_embeds.dtype,
        )
        return hidden_states, attn_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor,
        visual_embeds: torch.Tensor,
        token_turn_ids: torch.Tensor,
    ):
        hidden_states, attention_mask_4d = self.build_input_sequence(
            ids=input_ids,
            mask=mask,
            visual_embeds=visual_embeds,
            token_turn_ids=token_turn_ids,
        )
        outputs = self.backbone.model(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask_4d,
            use_cache=False,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state

        base_len = input_ids.shape[1]
        world_start = base_len
        world_end = world_start + self.num_world_query_tokens
        ego_start = world_end
        ego_end = ego_start + self.num_ego_query_tokens

        states_world = last_hidden[:, world_start:world_end, :]
        states_ego = last_hidden[:, ego_start:ego_end, :]
        return states_world, states_ego

    def get_trainable_params_info(self) -> Dict[str, Union[int, float]]:
        trainable, total = 0, 0
        for p in self.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return {
            "trainable": trainable,
            "total": total,
            "ratio": round(trainable / total * 100, 4),
        }
