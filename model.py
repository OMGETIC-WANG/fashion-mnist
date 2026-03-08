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


class ResLinear(nnx.Module):
    def __init__(
        self,
        features: int,
        activation: T.Callable[[jax.Array], jax.Array],
        *,
        use_batchnorm: bool = False,
        use_dropout: bool = False,
        dropout_rate: float = 0.1,
        rngs: nnx.Rngs,
    ):
        self.linear = nnx.Linear(features, features, rngs=rngs)
        self.activation = activation

        self.use_batchnorm = use_batchnorm
        if use_batchnorm:
            self.batchnorm = nnx.BatchNorm(features, rngs=rngs)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        y = jax.lax.cond(
            self.use_batchnorm,
            lambda inst, x: inst.batchnorm(x),
            lambda inst, x: x,
            self,
            self.linear(x),
        )
        y = self.activation(y)
        y = jax.lax.cond(
            self.use_dropout, lambda inst, x: inst.dropout(x), lambda inst, x: x, self, y
        )
        return x + y


class Sequential(nnx.Module):
    def __init__(self, layers: T.Sequence[T.Callable[[jax.Array], jax.Array]]):
        self.layers = nnx.List(layers)

    def __call__(self, x: jax.Array) -> jax.Array:
        return ApplyTrain(self.layers, x)


class PreCNN(nnx.Module):
    def __init__(self, model_features: int, *, rngs: nnx.Rngs, dropout_rate: float = 0):
        self.layers = nnx.List[T.Callable[[jax.Array], jax.Array]]([
            nnx.Conv(1, model_features // 2, (3, 3), padding="VALID", rngs=rngs),
            nnx.BatchNorm(model_features // 2, rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(model_features // 2, model_features, (3, 3), padding="VALID", rngs=rngs),
            nnx.BatchNorm(model_features, rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(model_features, model_features, (2, 2), (2, 2), padding="VALID", rngs=rngs),
            nnx.BatchNorm(model_features, rngs=rngs),
            nnx.leaky_relu,
        ])
        if dropout_rate > 0:
            self.layers.append(nnx.Dropout(dropout_rate, rngs=rngs))
        self.model_features = model_features

    def __call__(self, x: jax.Array) -> jax.Array:
        return ApplyTrain(self.layers, x).reshape(x.shape[0], -1, self.model_features)


class PreCNN2(nnx.Module):
    def __init__(self, model_features: int, *, rngs: nnx.Rngs, dropout_rate: float = 0):
        cnn_features = model_features // 2
        self.cnn1 = Sequential([
            nnx.Conv(1, cnn_features // 2, (3, 3), padding="SAME", rngs=rngs),
            nnx.BatchNorm(cnn_features // 2, rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(cnn_features // 2, cnn_features, (3, 3), padding="SAME", rngs=rngs),
            nnx.BatchNorm(cnn_features, rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(cnn_features, cnn_features, (2, 2), (2, 2), padding="VALID", rngs=rngs),
            nnx.BatchNorm(cnn_features, rngs=rngs),
            nnx.leaky_relu,
        ])
        self.cnn2 = Sequential([
            nnx.Conv(1, cnn_features // 2, (5, 5), padding="SAME", rngs=rngs),
            nnx.BatchNorm(cnn_features // 2, rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(cnn_features // 2, cnn_features, (5, 5), padding="SAME", rngs=rngs),
            nnx.BatchNorm(cnn_features, rngs=rngs),
            nnx.leaky_relu,
            nnx.Conv(cnn_features, cnn_features, (2, 2), (2, 2), padding="VALID", rngs=rngs),
            nnx.BatchNorm(cnn_features, rngs=rngs),
            nnx.leaky_relu,
        ])
        self.dropout_rate = dropout_rate
        if dropout_rate > 0:
            self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.model_features = model_features

    def __call__(self, x: jax.Array):
        x = jnp.concatenate([self.cnn1(x), self.cnn2(x)], axis=-1)
        x = nnx.cond(
            self.dropout_rate != 0, lambda inst, x: inst.dropout(x), lambda inst, x: x, self, x
        )
        return x.reshape(x.shape[0], -1, self.model_features)


class TransformerBlock(nnx.Module):
    def __init__(self, in_features: int, num_heads: int, dropout_rate: float, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads,
            in_features,
            use_bias=False,
            rngs=rngs,
            kernel_init=nnx.initializers.xavier_uniform(),
            decode=False,
        )
        self.fnn = MLP(in_features, in_features, [in_features * 4], nnx.gelu, rngs=rngs)

        self.pre_attention_norm = nnx.LayerNorm(in_features, rngs=rngs)
        self.pre_fnn_norm = nnx.LayerNorm(in_features, rngs=rngs)

        self.fnn_dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def __call__(self, x: jax.Array):
        x = x + self.attention(self.pre_attention_norm(x))
        y = self.fnn(self.pre_fnn_norm(x))
        return x + self.fnn_dropout(y)


class MnistModel(nnx.Module):
    def __init__(
        self,
        model_features: int,
        num_heads: int,
        num_encoder: int,
        rngs: nnx.Rngs,
    ):
        self.cnn = PreCNN2(model_features, rngs=rngs, dropout_rate=0.4)
        self.encoders = nnx.List([
            TransformerBlock(model_features, num_heads, 0.4, rngs) for _ in range(num_encoder)
        ])

        _, seqlen, _ = nnx.eval_shape(lambda m, x: m(x), self.cnn, jnp.zeros((1, 28, 28, 1))).shape
        self.pos_embedding = nnx.Param(
            nnx.initializers.normal(stddev=0.02)(rngs.params(), (seqlen, model_features))
        )

        dropout_rate = 0.4
        self.target_logits_mlp = Sequential([
            nnx.Linear(model_features, 256, rngs=rngs),
            nnx.BatchNorm(256, rngs=rngs),
            nnx.gelu,
            nnx.Dropout(dropout_rate, rngs=rngs),
            ResLinear(
                256,
                nnx.gelu,
                rngs=rngs,
                use_batchnorm=True,
                use_dropout=True,
                dropout_rate=dropout_rate,
            ),
            ResLinear(
                256,
                nnx.gelu,
                rngs=rngs,
                use_batchnorm=True,
                use_dropout=True,
                dropout_rate=dropout_rate,
            ),
            nnx.Linear(256, 10, rngs=rngs),
        ])

        self.model_features = model_features

    def __call__(self, x: jax.Array):
        x = self.cnn(x)
        batch_size, input_seq_len, _ = x.shape
        x += self.pos_embedding[:input_seq_len][None, ...]

        for encoder in self.encoders:
            x = encoder(x)

        x = jnp.mean(x, axis=1)
        return self.target_logits_mlp(x)
