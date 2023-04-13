import keras
import tensorflow as tf
from keras import layers

class VIT(keras.Model):
    def __init__(self,
                 image_size, 
                 patch_size,
                 n_patches,
                 projection_dim,
                 latent_dim, 
                 num_heads, 
                 head_size, 
                 n_blocks,
                 num_classes,
                 dropout_rate,
                 mlp_head_units,
                 x_train):
        
        self.patches = Patches(patch_size)
        self.patch_encoder = PatchEncoder(n_patches, projection_dim)
        self.blocks = [TransformerBlock(latent_dim, num_heads, head_size)
                        for block in n_blocks]
        self.classifier = layers.Dense(num_classes)

        self.data_augmentation = keras.Sequential([layers.Normalization(),
                                        layers.Resizing(image_size, image_size),
                                        layers.RandomFlip("horizontal"),
                                        layers.RandomRotation(factor=0.02),
                                        layers.RandomZoom(height_factor=0.2, width_factor=0.2)])
        
        # Compute the mean and the variance of the training data for normalization.
        self.data_augmentation.layers[0].adapt(x_train)
        self.dropout = dropout_rate
        self.mlp_head_units = mlp_head_units
    
        
    def call(self, inputs):
        augmented = self.data_augmentation(inputs)
        patches = self.patches(augmented)
        encoded_patches = self.patch_encoder(patches)
        
        for block in self.blocks:
            encoded_patches = block(encoded_patches)
            
        representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = layers.Flatten()(representation)
        representation = layers.Dropout(0.5)(representation)

        features = self.mlp(representation, 
                            hidden_units=self.mlp_head_units, 
                            dropout_rate=self.dropout)
        
        return self.classifier(features)


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(input_dim=num_patches,
                                                   output_dim=projection_dim)

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(images=images,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID")
        
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
    

class TransformerBlock(keras.layers.Layer):
    def __init__(self, dim, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = Attention(num_heads, head_size)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = Attention(num_heads, head_size)
        self.norm3 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = keras.layers.Dense(dim)

    def call(self, inputs):
        inputs, context = inputs
        x = self.attn1([self.norm1(inputs), None]) + inputs
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class Attention(keras.layers.Layer):
    def __init__(self, num_heads, head_size, **kwargs):
        super().__init__(**kwargs)
        self.to_q = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_k = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.to_v = keras.layers.Dense(num_heads * head_size, use_bias=False)
        self.scale = head_size**-0.5
        self.num_heads = num_heads
        self.head_size = head_size
        self.out_proj = keras.layers.Dense(num_heads * head_size)

    def td_dot(self, a, b):
        aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
        bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
        cc = keras.backend.batch_dot(aa, bb)
        return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))

    def call(self, inputs):
        inputs, context = inputs
        context = inputs if context is None else context
        q, k, v = self.to_q(inputs), self.to_k(context), self.to_v(context)
        q = tf.reshape(q, (-1, inputs.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = tf.transpose(q, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)
        k = tf.transpose(k, (0, 2, 3, 1))  # (bs, num_heads, head_size, time)
        v = tf.transpose(v, (0, 2, 1, 3))  # (bs, num_heads, time, head_size)

        score = self.td_dot(q, k) * self.scale
        weights = keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attn = self.td_dot(weights, v)
        attn = tf.transpose(attn, (0, 2, 1, 3))  # (bs, time, num_heads, head_size)
        out = tf.reshape(attn, (-1, inputs.shape[1], self.num_heads * self.head_size))
        return self.out_proj(out)



class GEGLU(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.dense = keras.layers.Dense(output_dim * 2)

    def call(self, inputs):
        x = self.dense(inputs)
        x, gate = x[..., : self.output_dim], x[..., self.output_dim :]
        tanh_res = keras.activations.tanh(
            gate * 0.7978845608 * (1 + 0.044715 * (gate**2))
        )
        return x * 0.5 * gate * (1 + tanh_res)