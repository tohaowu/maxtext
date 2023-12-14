"""
 Copyright 2023 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

""" Tests for Attention """
import jax
import unittest
import jax.numpy as jnp
import max_utils
from jax.sharding import Mesh
from layers import attentions
from jax.sharding import PartitionSpec as P
import numpy as np

import pyconfig
import sys

FlashMultiHeadDotProductAttention = attentions.FlashMultiHeadDotProductAttention
MultiHeadDotProductAttention = attentions.MultiHeadDotProductAttention


class AttentionTest(unittest.TestCase):
  """Test for the Attention """

  def setUp(self):
    super().setUp()

    pyconfig.initialize(
        sys.argv + ['third_party/py/maxtext/configs/base.yml'],
        per_device_batch_size=1.0,
        run_name='test',
        enable_checkpointing=False)
    self.cfg = pyconfig.config
    self.rng = jax.random.PRNGKey(0)

    devices_array = max_utils.create_device_mesh(self.cfg)
    self.mesh = Mesh(devices_array, self.cfg.mesh_axes)

    self.global_batch_size = self.cfg.global_batch_size_to_train_on
    self.num_heads = self.cfg.base_num_heads
    self.max_target_length = self.cfg.max_target_length
    self.head_dim = self.cfg.head_dim
    self.embed_dim = self.cfg.base_emb_dim

  def get_decoder_mask(self):
    a = jnp.stack([
        jnp.tri(self.max_target_length, dtype=self.dtype)[jnp.newaxis, :]
        for _ in range(self.global_batch_size)
    ])
    return a

  def get_data(self):
    lnx = jax.random.uniform(
        self.rng,
        shape=(self.global_batch_size, self.max_target_length, self.embed_dim),
        dtype=self.dtype,
    )
    decoder_segment_ids = jnp.ones(
        shape=(self.global_batch_size, self.max_target_length), dtype=np.int32
    )

    def batch_positions():
      return [
          jnp.arange(self.max_target_length, dtype=jnp.int32)
          for _ in range(self.global_batch_size)
      ]

    if self.global_batch_size > 1:
      decoder_positions = jnp.stack(batch_positions())

    decoder_mask = self.get_decoder_mask()
    return lnx, decoder_mask, decoder_segment_ids, decoder_positions

  def test_flash_mha_attention(self):
    """Test MHA layer and Flash MHA equivalence."""

    mha_attention_layer = MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        mesh=self.mesh,
        dtype=self.cfg.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
    )
    variable = mha_attention_layer.init(
        {'params': self.rng, 'aqt': self.rng},
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        jnp.ones(
            (self.global_batch_size, self.max_target_length, self.embed_dim)),
        'mha',
    )

    flash_attention_layer = FlashMultiHeadDotProductAttention(
        num_heads=self.num_heads,
        head_dim=self.head_dim,
        mesh=self.mesh,
        dtype=self.cfg.dtype,
        dropout_rate=self.cfg.dropout_rate,
        name='self_attention',
        max_target_length=self.max_target_length)

    lnx, decoder_mask, decoder_segment_ids, decoder_positions = self.get_data()
    bias = None
    mha_output = mha_attention_layer.apply(
        variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        mask=decoder_mask,
        bias=bias,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )
    flash_output = flash_attention_layer.apply(
        variable,
        lnx,
        lnx,
        decoder_segment_ids=decoder_segment_ids,
        inputs_positions=decoder_positions,
        mask=decoder_mask,
        bias=bias,
        deterministic=True,
        decode=False,
        rngs={'aqt': self.rng},
    )

    self.assertTrue(
        jax.numpy.allclose(
            flash_output, mha_output, rtol=1e-01, atol=1e-01, equal_nan=False
        )
    )


if __name__ == '__main__':
  unittest.main()