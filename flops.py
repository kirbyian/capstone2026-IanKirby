"""Computes the flops needed for training/running transformer networks."""

"""Partial code is from https://github.com/soarsmu/Compressor/blob/main/flops.py"""

 # random number, >=, multiply activations by dropout mask, multiply activations
# by correction (1 / (1 - dropout_rate))
DROPOUT_FLOPS = 4

 # compute mean activation (sum), computate variance of activation
 # (square and sum), bias (add), scale (multiply)
LAYER_NORM_FLOPS = 5

# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8

# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5

class TransformerHparams(object):
    """Computes the train/inference FLOPs for encoder–decoder transformers."""

    def __init__(self, d_model=768, num_encoder_layers=12, num_decoder_layers=12, s_enc=512, s_dec=128, v=32000, i=3072, heads=12, tie_embeddings=True,encoder_ffn_dim=1024,decoder_ffn_dim=1024):
        self.num_encoder_layers = num_encoder_layers  # encoder layers
        self.num_decoder_layers = num_decoder_layers  # decoder layers
        self.s_enc = s_enc  # encoder sequence length
        self.s_dec = s_dec  # decoder sequence length
        self.v = v  # vocab size
        self.d_model = d_model  # embedding size
        self.heads = heads
        self.kqv = d_model
        self.tie_embeddings = tie_embeddings
        self.encoder_ffn_dim = encoder_ffn_dim
        self.decoder_ffn_dim = decoder_ffn_dim

    def get_encoder_block_flops(self):
        """Computes the FLOPs for a single encoder layer."""
        block_flops = dict(
            kqv=3 * 2 * self.d_model * self.kqv,
            kqv_bias=3 * self.kqv,
            attention_scores=2 * self.kqv * self.s_enc,
            attn_softmax=SOFTMAX_FLOPS * self.s_enc * self.d_model,
            attention_dropout=DROPOUT_FLOPS * self.s_enc * self.d_model,
            attention_scale=self.s_enc * self.d_model,
            attention_weighted_avg_values=2 * self.d_model * self.s_enc,
            attn_output=2 * self.d_model * self.d_model,
            attn_output_bias=self.d_model,
            attn_output_dropout=DROPOUT_FLOPS * self.d_model,
            attn_output_residual=self.d_model,
            attn_output_layer_norm=LAYER_NORM_FLOPS,
            intermediate=2 * self.d_model * self.encoder_ffn_dim,
            intermediate_act=ACTIVATION_FLOPS * self.encoder_ffn_dim,
            intermediate_bias=self.encoder_ffn_dim,
            output=2 * self.d_model * self.encoder_ffn_dim,
            output_bias=self.d_model,
            output_dropout=DROPOUT_FLOPS * self.d_model,
            output_residual=self.d_model,
            output_layer_norm=LAYER_NORM_FLOPS * self.d_model
        )
        return sum(block_flops.values()) * self.s_enc

    def get_decoder_block_flops(self):
        """Computes the FLOPs for a single decoder layer."""
        block_flops = dict(
            kqv=3 * 2 * self.d_model * self.kqv,
            kqv_bias=3 * self.kqv,
            attention_scores=2 * self.kqv * self.s_dec,
            attn_softmax=SOFTMAX_FLOPS * self.s_dec * self.d_model,
            attention_dropout=DROPOUT_FLOPS * self.s_dec * self.d_model,
            attention_scale=self.s_dec * self.d_model,
            attention_weighted_avg_values=2 * self.d_model * self.s_dec,
            attn_output=2 * self.d_model * self.d_model,
            attn_output_bias=self.d_model,
            attn_output_dropout=DROPOUT_FLOPS * self.d_model,
            attn_output_residual=self.d_model,
            attn_output_layer_norm=LAYER_NORM_FLOPS,
            intermediate=2 * self.d_model * self.decoder_ffn_dim,
            intermediate_act=ACTIVATION_FLOPS * self.decoder_ffn_dim,
            intermediate_bias=self.decoder_ffn_dim,
            output=2 * self.d_model * self.decoder_ffn_dim,
            output_bias=self.d_model,
            output_dropout=DROPOUT_FLOPS * self.d_model,
            output_residual=self.d_model,
            output_layer_norm=LAYER_NORM_FLOPS * self.d_model
        )
        return sum(block_flops.values()) * self.s_dec

    def get_embedding_flops(self):
        flops = dict(
            pos_emb=2 * self.d_model * (self.s_enc + self.s_dec),
            emb_layer_norm=LAYER_NORM_FLOPS * self.d_model,
            emb_dropout=DROPOUT_FLOPS * self.d_model
        )
        return sum(flops.values()) * (self.s_enc + self.s_dec)

    def get_cross_attention_flops(self):
        return (
            # Q from decoder states
                2 * self.s_dec * self.d_model * self.d_model

                # K and V from encoder states
                + 2 * self.s_enc * self.d_model * self.d_model
                + 2 * self.s_enc * self.d_model * self.d_model

                # attention scores: QK^T
                + 2 * self.s_dec * self.s_enc * self.d_model

                # softmax/dropout/scale
                + SOFTMAX_FLOPS * self.s_dec * self.s_enc * self.d_model
                + DROPOUT_FLOPS * self.s_dec * self.s_enc * self.d_model
                + self.s_dec * self.s_enc * self.d_model

                # attention probabilities × V
                + 2 * self.s_dec * self.s_enc * self.d_model

                # output projection
                + 2 * self.s_dec * self.d_model * self.d_model
        )

    def get_lm_head_flops(self):
        return 2 * self.s_dec * self.d_model * self.v

    def get_infer_flops(self):
        return (
                self.num_encoder_layers * self.get_encoder_block_flops()
                + self.num_decoder_layers * self.get_decoder_block_flops()
                + self.num_decoder_layers * self.get_cross_attention_flops()
                + self.get_embedding_flops()
                + self.get_lm_head_flops()
        )

    def get_params(self):
        embedding_params = self.v * self.d_model + (self.s_enc + self.s_dec) * self.d_model
        encoder_block = (
        # self-attention: Q, K, V, output projection
        4 * self.d_model * self.d_model
        # FFN: h -> i and i -> h
        + 2 * self.d_model * self.encoder_ffn_dim
        + 4 * self.d_model) * self.num_encoder_layers
        decoder_block = (
        # decoder self-attention
        4 * self.d_model * self.d_model
        # cross-attention: Q, K, V, output projection
        + 4 * self.d_model * self.d_model
        # FFN
        + 2 * self.d_model * self.encoder_ffn_dim
        # layernorm/bias terms
        + 6 * self.d_model) * self.num_decoder_layers
        output_proj = 0 if self.tie_embeddings else self.v * self.d_model
        return embedding_params + encoder_block + decoder_block + output_proj

