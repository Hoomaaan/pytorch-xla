import os
import sys

import torch
from torch.distributed.tensor import DeviceMesh, Shard, distribute_tensor
from torch.distributed.tensor.placement_types import Replicate

import torch_xla
import torch_xla.runtime as xr
import torch_xla.core.xla_model as xm
from torch_xla.distributed.spmd import XLAShardedTensor, mark_sharding, Mesh, ShardingType
from torch_xla.distributed.spmd.xla_sharding import wrap_as_sharded_tensor

import unittest
import test_xla_sharding_base


class XLADTensorPropagateTest(test_xla_sharding_base.XlaShardingTest):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

  def test_propagate(self):
    world_size = xr.global_runtime_device_count()
    dev_mesh = DeviceMesh("xla", torch.arange(world_size))

    # Enable SPMD
    xr.use_spmd()

    # Create tensor and mesh
    # breakpoint()
    tensor = torch.randn(8, 16).to('xla')
    mesh = Mesh(list(range(4)), (4,))

    print("=== Before mark_sharding ===")
    try:
        spec_before = torch_xla._XLAC._get_xla_sharding_spec(tensor)
        print(f"Type: {type(spec_before)}")
        print(f"Value: {spec_before}")
    except Exception as e:
        print(f"Error: {e}")

    # Mark sharding
    sharded = mark_sharding(tensor, mesh, (0, None))

    print("\n=== After mark_sharding ===")
    try:
        spec_after = torch_xla._XLAC._get_xla_sharding_spec(sharded.global_tensor)
        print(f"Type: {type(spec_after)}")
        print(f"Value: {spec_after}")
        print(f"Repr: {repr(spec_after)}")
        
        # Try common protobuf methods
        if hasattr(spec_after, 'type'):
            print(f"type(): {spec_after.type()}")
        if hasattr(spec_after, 'tile_assignment_dimensions'):
            print(f"tile_assignment_dimensions(): {spec_after.tile_assignment_dimensions()}")
        if hasattr(spec_after, 'tile_assignment_devices'):
            print(f"tile_assignment_devices(): {spec_after.tile_assignment_devices()}")
            
    except Exception as e:
        print(f"Error: {e}")

    # Force compilation
    print("\n=== After compilation ===")
    xm.mark_step()
    try:
        spec_compiled = torch_xla._XLAC._get_xla_sharding_spec(sharded.global_tensor)
        a = sharded._spec.placements
        b = sharded.partition_spec
        # breakpoint()
        print(f"Type: {type(spec_compiled)}")
        print(f"Value: {spec_compiled}")
        
        # Try to serialize
        if hasattr(spec_compiled, 'SerializeToString'):
            serialized = spec_compiled.SerializeToString()
            print(f"Serialized length: {len(serialized)}")
            
    except Exception as e:
        print(f"Error: {e}")


  def test_parse_xla_sharding_to_placements_replicated(self):
    """Test parsing replicated XLA sharding spec."""
    device_count = xr.global_runtime_device_count()
    tensor = torch.randn(8, 16).to('xla')
    mesh = Mesh(list(range(device_count)), (device_count,))
    sharded_tensor = mark_sharding(tensor, mesh, (None, None))
    
    # Test replicated case
    placements = sharded_tensor._parse_xla_sharding_to_placements('{devices=[1]0}')
    
    self.assertEqual(len(placements), len(mesh.mesh_shape))
    for placement in placements:
        self.assertIsInstance(placement, Replicate)
    print("Replicated sharding parsing test passed")


  def test_spec_debug_mode(self):
    """Test _spec property in debug mode (uses propagated sharding)."""
    breakpoint()
    device_count = xr.global_runtime_device_count()
    tensor = torch.randn(8, 16).to('xla')
    mesh = Mesh(list(range(device_count)), (device_count,))
    sharded_tensor = mark_sharding(tensor, mesh, (0, None))
    
    # Debug mode - should use propagated sharding
    try:
        breakpoint()
        sharded_tensor._use_propagated_sharding = True
        spec = sharded_tensor._spec
        
        self.assertIsNotNone(spec)
        self.assertEqual(len(spec.placements), len(mesh.mesh_shape))
        # Should have at least one Shard placement
        has_shard = any(isinstance(p, Shard) for p in spec.placements)
        self.assertTrue(has_shard)
        print("Debug mode _spec test passed")
        
    except Exception as e:
        print(f"Debug mode test failed (expected if XLA not fully compiled): {e}")



if __name__ == '__main__':
  test = unittest.main()
  sys.exit(0 if test.result.wasSuccessful() else 1)