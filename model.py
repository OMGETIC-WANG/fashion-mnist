import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import typing as T


def ApplyTrain(layers: T.Sequence[T.Callable[[jax.Array], jax.Array]], x: jax.Array) -> jax.Array:
    for layer in layers:
        x = layer(x)
    return x


class MLP(nnx.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: T.Sequence[int],
        activation: T.Callable[[jax.Array], jax.Array],
        rngs: nnx.Rngs,
    ):
        self.layers = nnx.List()
        self.activation = activation

        features = [in_features] + list(hidden_features) + [out_features]
        for din, dout in zip(features[:-1], features[1:]):
            self.layers.append(nnx.Linear(din, dout, rngs=rngs))

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class PreCNN(nnx.Module):
    def __init__(self, model_features: int, rngs: nnx.Rngs):
        self.layers = nnx.List[T.Callable[[jax.Array], jax.Array]]([
            nnx.Conv(1, model_features // 2, (3, 3), padding="VALID", rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(model_features // 2, model_features, (3, 3), padding="VALID", rngs=rngs),
            nnx.leaky_relu,
        ])
        self.model_features = model_features

    def __call__(self, x: jax.Array) -> jax.Array:
        return ApplyTrain(self.layers, x).reshape(x.shape[0], -1, self.model_features)


class TransformerBlock(nnx.Module):
    def __init__(self, in_features: int, num_heads: int, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads,
            in_features,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            decode=False,
        )
        self.fnn = MLP(
            in_features, in_features, [in_features * 4, in_features * 4], nnx.gelu, rngs=rngs
        )

        self.pre_attention_norm = nnx.LayerNorm(in_features, rngs=rngs)
        self.pre_fnn_norm = nnx.LayerNorm(in_features, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = x + self.attention(self.pre_attention_norm(x))
        y = self.fnn(self.pre_fnn_norm(x))
        return x + y


class MnistModel(nnx.Module):
    def __init__(
        self,
        model_features: int,
        num_encoder: int,
        rngs: nnx.Rngs,
    ):
        self.cnn = PreCNN(model_features, rngs)
        self.encoders = nnx.List([
            TransformerBlock(model_features, 4, rngs) for _ in range(num_encoder)
        ])

        _, seqlen, _ = jax.eval_shape(lambda: self.cnn(jnp.zeros((1, 28, 28, 1)))).shape
        self.pos_embedding = nnx.Param(
            nnx.initializers.uniform()(rngs.params(), (seqlen, model_features))
        )
        self.seq_elem_weights = nnx.Param(
            nnx.initializers.uniform()(rngs.params(), (seqlen, model_features))
        )
        self.target_logits_mlp = MLP(seqlen, 10, [seqlen * 4, seqlen * 4], nnx.leaky_relu, rngs)

        self.model_features = model_features

    def __call__(self, x: jax.Array):
        x = self.cnn(x)
        batch_size, input_seq_len, _ = x.shape
        x += self.pos_embedding[:input_seq_len][None, ...]

        for encoder in self.encoders:
            x = encoder(x)

        x = jnp.einsum("bsf,sf->bs", x, self.seq_elem_weights)
        return self.target_logits_mlp(x)
