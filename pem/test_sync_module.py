#!/usr/bin/env python3
"""
Test script for PEM Sync Module.

Usage:
    python -m pem.test_sync_module
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_context_encoder():
    """Test ContextEncoder aggregates history correctly."""
    from pem.sync_module import ContextEncoder

    logger.info("\n" + "=" * 60)
    logger.info("TEST: ContextEncoder")
    logger.info("=" * 60)

    d_model = 64
    encoder = ContextEncoder(d_model=d_model, n_heads=4)
    encoder.eval()

    # Input: history (B, S, T, D)
    B, S, T, D = 2, 16, 8, d_model
    history = torch.randn(B, S, T, D)

    logger.info(f"Input shape: {history.shape}")

    with torch.no_grad():
        context = encoder(history)

    logger.info(f"Output shape: {context.shape}")

    assert context.shape == (B, S, D), f"Expected {(B, S, D)}, got {context.shape}"

    logger.info("ContextEncoder test PASSED")
    return True


def test_change_point_detector():
    """Test ChangePointDetector detects boundaries."""
    from pem.sync_module import ChangePointDetector

    logger.info("\n" + "=" * 60)
    logger.info("TEST: ChangePointDetector")
    logger.info("=" * 60)

    d_model = 64
    cpd = ChangePointDetector(d_model=d_model, window_size=4, threshold=0.3)
    cpd.eval()

    B, S, D = 2, 32, d_model

    # Create features with clear change points
    # First half: one distribution, second half: different
    features = torch.randn(B, S, D)
    features[:, S//2:, :] = features[:, S//2:, :] + 5.0  # Shift second half

    logger.info(f"Input shape: {features.shape}")

    with torch.no_grad():
        boundary_probs, chunk_ids = cpd(features)

    logger.info(f"Boundary probs shape: {boundary_probs.shape}")
    logger.info(f"Chunk IDs shape: {chunk_ids.shape}")
    logger.info(f"Boundary probs around midpoint: {boundary_probs[0, S//2-2:S//2+2].tolist()}")
    logger.info(f"Number of unique chunks: {chunk_ids[0].unique().shape[0]}")

    assert boundary_probs.shape == (B, S)
    assert chunk_ids.shape == (B, S)
    assert boundary_probs.min() >= 0 and boundary_probs.max() <= 1

    # Should detect at least one boundary (around the shift)
    assert chunk_ids[0].max() >= 1, "Should detect at least one boundary"

    logger.info("ChangePointDetector test PASSED")
    return True


def test_memory_bank_write_read():
    """Test MemoryBank write and read operations."""
    from pem.sync_module import MemoryBank

    logger.info("\n" + "=" * 60)
    logger.info("TEST: MemoryBank write/read")
    logger.info("=" * 60)

    memory_dim = 64
    memory = MemoryBank(num_slots=10, memory_dim=memory_dim, n_heads=4)

    B, S, D = 1, 16, memory_dim

    # Create features and chunk IDs
    features = torch.randn(B, S, D)
    chunk_ids = torch.zeros(B, S, dtype=torch.long)
    chunk_ids[:, 8:] = 1  # Two chunks

    # Optional surprise
    surprise = torch.rand(B, S)

    logger.info(f"Features shape: {features.shape}")
    logger.info(f"Chunk IDs: {chunk_ids[0].tolist()}")
    logger.info(f"Initial occupied slots: {memory.occupied.sum().item()}")

    # Write to memory
    memory.write(features, chunk_ids, surprise)

    logger.info(f"After write, occupied slots: {memory.occupied.sum().item()}")

    # Read from memory
    query = torch.randn(B, S, D)
    retrieved = memory.read(query)

    logger.info(f"Retrieved shape: {retrieved.shape}")

    assert retrieved.shape == (B, S, D)
    assert memory.occupied.sum() >= 2, "Should have stored at least 2 chunks"

    logger.info("MemoryBank write/read test PASSED")
    return True


def test_memory_bank_eviction():
    """Test MemoryBank importance-weighted eviction."""
    from pem.sync_module import MemoryBank

    logger.info("\n" + "=" * 60)
    logger.info("TEST: MemoryBank eviction")
    logger.info("=" * 60)

    memory = MemoryBank(num_slots=5, memory_dim=32, n_heads=4)

    B, S, D = 1, 8, 32

    # Fill memory with low-surprise chunks
    for i in range(5):
        features = torch.randn(B, S, D)
        chunk_ids = torch.zeros(B, S, dtype=torch.long)
        surprise = torch.full((B, S), 0.1)  # Low surprise
        memory.write(features, chunk_ids, surprise)

    logger.info(f"Memory full, occupied: {memory.occupied.sum().item()}")
    logger.info(f"Importances: {memory.importance.tolist()}")

    # Write a high-surprise chunk
    features_important = torch.randn(B, S, D)
    chunk_ids = torch.zeros(B, S, dtype=torch.long)
    surprise_high = torch.full((B, S), 0.9)  # High surprise
    memory.write(features_important, chunk_ids, surprise_high)

    logger.info(f"After high-surprise write, importances: {memory.importance.tolist()}")

    # The high-surprise chunk should have replaced a low-importance one
    assert memory.importance.max() > 0.5, "High-surprise chunk should have high importance"

    logger.info("MemoryBank eviction test PASSED")
    return True


def test_personality_module():
    """Test PersonalityModule forward and seeding."""
    from pem.sync_module import PersonalityModule

    logger.info("\n" + "=" * 60)
    logger.info("TEST: PersonalityModule")
    logger.info("=" * 60)

    d_model = 64
    personality = PersonalityModule(d_model=d_model, hidden_dim=32)

    B, S, D = 2, 16, d_model
    context = torch.randn(B, S, D)

    logger.info(f"Context shape: {context.shape}")
    logger.info(f"Initial personality norm: {personality.base_personality.norm().item():.4f}")

    # Forward pass
    with torch.no_grad():
        signal = personality(context)

    logger.info(f"Output shape: {signal.shape}")
    assert signal.shape == (B, S, D)

    # Test seeding from text
    text_embed = torch.randn(D)
    old_personality = personality.base_personality.clone()
    personality.seed_from_text(text_embed)

    logger.info(f"After seeding, personality changed: {(personality.base_personality != old_personality).any().item()}")
    assert not torch.equal(personality.base_personality, old_personality), "Personality should change after seeding"

    logger.info("PersonalityModule test PASSED")
    return True


def test_sync_module_forward():
    """Test full SyncModule forward pass."""
    from pem.sync_module import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: SyncModule forward")
    logger.info("=" * 60)

    config = SyncModuleConfig(
        d_model=64,
        sync_pairs=64,
        n_heads=4,
        memory_slots=20,
        cpd_window_size=4,
    )

    module = SyncModule(config)
    module.eval()

    # Input: history (B, S, T, D)
    B, S, T, D = 2, 32, 8, config.d_model
    history = torch.randn(B, S, T, D)

    logger.info(f"Input shape: {history.shape}")

    with torch.no_grad():
        sync = module(history)

    logger.info(f"Output shape: {sync.shape}")
    logger.info(f"Sync range: [{sync.min():.3f}, {sync.max():.3f}]")
    logger.info(f"Memory occupied: {module.memory.occupied.sum().item()}")

    assert sync.shape == (B, S, config.sync_pairs)

    logger.info("SyncModule forward test PASSED")
    return True


def test_sync_module_with_surprise():
    """Test SyncModule with surprise signal."""
    from pem.sync_module import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: SyncModule with surprise")
    logger.info("=" * 60)

    config = SyncModuleConfig(
        d_model=64,
        sync_pairs=64,
        n_heads=4,
        memory_slots=20,
    )

    module = SyncModule(config)
    module.eval()

    B, S, T, D = 2, 32, 8, config.d_model
    history = torch.randn(B, S, T, D)
    surprise = torch.rand(B, S, 1)  # Surprise magnitude from PEM

    logger.info(f"History shape: {history.shape}")
    logger.info(f"Surprise shape: {surprise.shape}")

    with torch.no_grad():
        sync = module(history, surprise=surprise)

    logger.info(f"Output shape: {sync.shape}")
    logger.info(f"Memory importances: {module.memory.importance[:10].tolist()}")

    assert sync.shape == (B, S, config.sync_pairs)

    logger.info("SyncModule with surprise test PASSED")
    return True


def test_gradient_flow():
    """Test that gradients flow through entire module."""
    from pem.sync_module import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Gradient flow")
    logger.info("=" * 60)

    config = SyncModuleConfig(
        d_model=64,
        sync_pairs=64,
        n_heads=4,
        memory_slots=10,
    )

    module = SyncModule(config)
    module.train()

    B, S, T, D = 1, 16, 4, config.d_model
    history = torch.randn(B, S, T, D, requires_grad=True)

    # Forward
    sync = module(history)

    # Backward
    loss = sync.mean()
    loss.backward()

    logger.info(f"History grad exists: {history.grad is not None}")
    logger.info(f"History grad norm: {history.grad.norm().item():.6f}")

    # Check gradients in module
    has_grads = False
    for name, param in module.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grads = True
            break

    logger.info(f"Module has gradients: {has_grads}")

    assert history.grad is not None
    assert has_grads

    logger.info("Gradient flow test PASSED")
    return True


def test_interface_compatibility():
    """Test that SyncModule has same interface as EnhancedSynchronizationModule."""
    from pem.sync_module import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Interface compatibility")
    logger.info("=" * 60)

    config = SyncModuleConfig(
        d_model=512,
        sync_pairs=512,
        n_heads=8,
    )

    module = SyncModule(config)
    module.eval()

    # Same interface as EnhancedSynchronizationModule
    B, S, T, D = 2, 128, 8, 512
    history = torch.randn(B, S, T, D)

    logger.info(f"Input: (B={B}, S={S}, T={T}, D={D})")

    with torch.no_grad():
        sync = module(history)

    logger.info(f"Output: {sync.shape}")

    assert sync.shape == (B, S, config.sync_pairs)
    assert sync.dtype == history.dtype

    logger.info("Interface compatibility test PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Sync Module Tests")
    logger.info("=" * 60)

    tests = [
        ("ContextEncoder", test_context_encoder),
        ("ChangePointDetector", test_change_point_detector),
        ("MemoryBank write/read", test_memory_bank_write_read),
        ("MemoryBank eviction", test_memory_bank_eviction),
        ("PersonalityModule", test_personality_module),
        ("SyncModule forward", test_sync_module_forward),
        ("SyncModule with surprise", test_sync_module_with_surprise),
        ("Gradient flow", test_gradient_flow),
        ("Interface compatibility", test_interface_compatibility),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
        except Exception as e:
            logger.error(f"\n{name} test FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    logger.info("\n" + "=" * 60)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 60)

    return failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
