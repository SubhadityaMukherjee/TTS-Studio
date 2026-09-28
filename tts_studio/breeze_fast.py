"""Optimized Breeze TTS 2 depth decoding for Apple Silicon (MLX).

Vendored and adapted from https://github.com/xzf-thu/BreezeTTS2_Mac_Streaming
(Apache License 2.0). mlx-audio 0.5.1's depth decoder recomputes each frame's
already-produced acoustic codes from scratch instead of reusing its KV cache,
plus syncs to the CPU once per acoustic code. ``FastDepth`` rewrites that loop
with intra-frame KV reuse and a single host read-back per frame, bringing the
real-time factor from ~4 down to ~1 with 4bit weights.

Only the generation-acceleration core is kept (no playback/benchmark harness);
the text encoder, backbone decoding, EOS handling, sampling, and the streaming
codec remain mlx-audio's release implementation.
"""

from __future__ import annotations


class FrameKVCache:
    """Small append-only cache, newly allocated for EACH acoustic frame.

    At most num_codebooks entries; no 256-token capacity allocation needed.
    Used with mlx-audio's Llama attention offset/update_and_fetch API.
    """

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        import mlx.core as mx

        if self.keys is None:
            self.keys, self.values = keys, values
        else:
            self.keys = mx.concatenate((self.keys, keys), axis=2)
            self.values = mx.concatenate((self.values, values), axis=2)
        self.offset += keys.shape[2]
        return self.keys, self.values


class FastDepth:
    """Cache depth attention and defer host reads to the end of each frame."""

    def __init__(self, model, mode="cached"):
        self.model, self.mode = model, mode
        self.functions = {}
        self.original = None

    def initial(self, first, hidden):
        import mlx.core as mx

        m = self.model.depth_decoder.model
        if m.backbone_hidden_state_projector is not None:
            hidden = m.backbone_hidden_state_projector(hidden)
        ids = mx.broadcast_to(first.reshape(1, 1), (hidden.shape[0], 1))
        embeds = mx.concatenate((hidden[:, None, :], m.embed_tokens(ids)), axis=1)
        x = m.inputs_embeds_projector(embeds)
        caches = [FrameKVCache() for _ in m.layers]
        for layer, cache in zip(m.layers, caches):
            x = layer(x, "causal", cache)
        return x, caches

    def advance(self, token, codebook_index, caches, batch):
        import mlx.core as mx

        m = self.model.depth_decoder.model
        ids = mx.broadcast_to(token.reshape(1, 1), (batch, 1))
        # At absolute depth position p>=1, upstream uses offset (p-1)*vocab.
        embeds = m.embed_tokens(ids + codebook_index * m.vocab_size)
        x = m.inputs_embeds_projector(embeds)
        for layer, cache in zip(m.layers, caches):
            x = layer(x, None, cache)
        return x

    def logits(self, x, head):
        d = self.model.depth_decoder
        return d.model.norm(x[:, -1, :]) @ d.codebooks_head.weight[head]

    def _function(self, temperature, top_p, top_k, cfg_scale, use_cfg):
        import mlx.core as mx
        import mlx.nn as nn
        from mlx_audio.lm.sample_utils import make_sampler

        key = (temperature, top_p, top_k, cfg_scale, use_cfg)
        if key in self.functions:
            return self.functions[key]
        valid = self.model.vocab_size
        effective_k = min(top_k, valid) if top_k else 0
        if effective_k == valid:
            effective_k = 0
        sample = make_sampler(temp=temperature, top_p=top_p, top_k=effective_k)

        def frame(first, hidden):
            x, caches = self.initial(first, hidden)
            tokens = [first.reshape(1)]
            for head in range(self.model.num_codebooks - 1):
                scores = self.logits(x, head)
                if use_cfg:
                    scores = scores[1:2] + cfg_scale * (scores[:1] - scores[1:2])
                scores = self.model._mask_reserved_codec_logits(scores)[..., :valid]
                token = (
                    sample(nn.log_softmax(scores, axis=-1)).astype(mx.int32).reshape(1)
                )
                tokens.append(token)
                if head + 1 < self.model.num_codebooks - 1:
                    x = self.advance(token, head + 1, caches, hidden.shape[0])
            return mx.concatenate(tokens)

        fn = (
            mx.compile(frame, inputs=mx.random.state, outputs=mx.random.state)
            if self.mode == "compiled"
            else frame
        )
        self.functions[key] = fn
        return fn

    def generate_frame(
        self,
        first_codebook,
        conditional_hidden,
        *,
        unconditional_hidden,
        cfg_scale,
        temperature,
        top_p,
        top_k,
    ):
        import mlx.core as mx

        if conditional_hidden.shape[0] != 1:
            raise ValueError("FastDepth currently supports one utterance at a time.")
        if self.model.num_codebooks != self.model.depth_decoder.model.num_codebooks:
            raise ValueError("Wrapper/depth codebook counts do not match.")
        hidden = conditional_hidden
        use_cfg = unconditional_hidden is not None
        if use_cfg:
            hidden = mx.concatenate((hidden, unconditional_hidden), axis=0)
        fn = self._function(temperature, top_p, top_k, cfg_scale, use_cfg)
        codes = fn(mx.array([first_codebook], dtype=mx.int32), hidden)
        # One host read for all remaining codebooks instead of .item() per code.
        return codes.tolist()

    def install(self):
        if self.mode == "native":
            return
        target = self.model
        cls = type(target)
        if getattr(cls._depth_tokens, "_fast_depth_owner", None) is not None:
            return
        original = cls._depth_tokens
        self.original = original
        owner = self

        def optimized(obj, *args, **kwargs):
            if obj is target:
                return owner.generate_frame(*args, **kwargs)
            return original(obj, *args, **kwargs)

        optimized._fast_depth_owner = self
        cls._depth_tokens = optimized

    def close(self):
        if self.original is not None:
            cls = type(self.model)
            if getattr(cls._depth_tokens, "_fast_depth_owner", None) is self:
                cls._depth_tokens = self.original
            self.original = None
