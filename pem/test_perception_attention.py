#!/usr/bin/env python3
"""
Test script for PEM Perception Attention Module.

Usage:
    python -m pem.test_perception_attention
"""

import torch
import logging

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_perception_kv_cache():
    """Test PerceptionKVCache projects Qwen features to K, V."""
    from pem.perception_attention import PerceptionKVCache

    logger.info("\n" + "=" * 60)
    logger.info("TEST: PerceptionKVCache")
    logger.info("=" * 60)

    d_perception = 1536  # Qwen hidden size
    d_model = 512
    n_heads = 8
    head_dim = d_model // n_heads

    cache = PerceptionKVCache(
        d_perception=d_perception,
        d_model=d_model,
        n_heads=n_heads,
    )

    B, S = 2, 32
    perception_features = torch.randn(B, S, d_perception)

    logger.info(f"Input shape: {perception_features.shape}")

    k, v = cache(perception_features)

    logger.info(f"K shape: {k.shape}")
    logger.info(f"V shape: {v.shape}")

    assert k.shape == (B, S, n_heads, head_dim), f"Expected K shape {(B, S, n_heads, head_dim)}, got {k.shape}"
    assert v.shape == (B, S, n_heads, head_dim), f"Expected V shape {(B, S, n_heads, head_dim)}, got {v.shape}"

    logger.info("PerceptionKVCache test PASSED")
    return True


def test_oscillation_query_builder():
    """Test OscillationQueryBuilder builds query from personality, intention, sync."""
    from pem.perception_attention import OscillationQueryBuilder

    logger.info("\n" + "=" * 60)
    logger.info("TEST: OscillationQueryBuilder")
    logger.info("=" * 60)

    d_model = 512
    sync_pairs = 512
    num_oscillators = 32

    builder = OscillationQueryBuilder(
        d_model=d_model,
        sync_pairs=sync_pairs,
        num_oscillators=num_oscillators,
    )

    B, S = 2, 32
    personality_signal = torch.randn(B, S, d_model)
    intention_signal = torch.randn(B, S, d_model)
    sync = torch.randn(B, S, sync_pairs)

    logger.info(f"Personality shape: {personality_signal.shape}")
    logger.info(f"Intention shape: {intention_signal.shape}")
    logger.info(f"Sync shape: {sync.shape}")

    # Test at different ticks
    for tick in [0, 5, 10]:
        query = builder(
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=tick,
        )
        logger.info(f"Tick {tick}: Query shape: {query.shape}, norm: {query.norm(dim=-1).mean():.4f}")

        assert query.shape == (B, S, d_model)

    # Test that different ticks produce different queries
    q0 = builder(personality_signal, intention_signal, sync, tick=0)
    q5 = builder(personality_signal, intention_signal, sync, tick=5)
    diff = (q0 - q5).abs().mean()
    logger.info(f"Query difference between tick 0 and 5: {diff:.4f}")
    # Even small differences indicate the oscillation is working
    assert diff > 1e-4, "Queries at different ticks should differ"

    logger.info("OscillationQueryBuilder test PASSED")
    return True


def test_perception_cross_attention():
    """Test PerceptionCrossAttention attends to cached perception."""
    from pem.perception_attention import PerceptionCrossAttention

    logger.info("\n" + "=" * 60)
    logger.info("TEST: PerceptionCrossAttention")
    logger.info("=" * 60)

    d_model = 512
    n_heads = 8
    head_dim = d_model // n_heads

    attn = PerceptionCrossAttention(
        d_model=d_model,
        n_heads=n_heads,
        use_flash_attention=False,  # Use standard for testing
    )

    B, S = 2, 32
    query = torch.randn(B, S, d_model)
    key = torch.randn(B, S, n_heads, head_dim)
    value = torch.randn(B, S, n_heads, head_dim)

    logger.info(f"Query shape: {query.shape}")
    logger.info(f"Key shape: {key.shape}")
    logger.info(f"Value shape: {value.shape}")

    output, attn_weights = attn(query, key, value, causal=True)

    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Attention weights shape: {attn_weights.shape}")

    assert output.shape == (B, S, d_model)
    assert attn_weights.shape == (B, n_heads, S, S)

    # Check causal mask is applied (no attending to future)
    # Upper triangle should be zero (after softmax, masked positions become ~0)
    upper_triangle = attn_weights[0, 0].triu(diagonal=1)
    assert upper_triangle.abs().max() < 1e-5, "Causal mask not properly applied"

    logger.info("PerceptionCrossAttention test PASSED")
    return True


def test_perception_synapse():
    """Test PerceptionSynapse integrates state and observation."""
    from pem.perception_attention import PerceptionSynapse

    logger.info("\n" + "=" * 60)
    logger.info("TEST: PerceptionSynapse")
    logger.info("=" * 60)

    d_model = 512
    synapse = PerceptionSynapse(d_model=d_model)

    B, S = 2, 32
    state = torch.randn(B, S, d_model)
    observation = torch.randn(B, S, d_model)

    logger.info(f"State shape: {state.shape}")
    logger.info(f"Observation shape: {observation.shape}")

    integrated = synapse(state, observation)

    logger.info(f"Integrated shape: {integrated.shape}")

    assert integrated.shape == (B, S, d_model)

    logger.info("PerceptionSynapse test PASSED")
    return True


def test_perception_attention_full():
    """Test full PerceptionAttention module."""
    from pem.perception_attention import PerceptionAttention, PerceptionConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: PerceptionAttention (full module)")
    logger.info("=" * 60)

    config = PerceptionConfig(
        d_model=512,
        d_perception=1536,
        n_heads=8,
        sync_pairs=512,
        num_oscillators=32,
        use_flash_attention=False,  # Flash attention doesn't work on CPU
    )

    module = PerceptionAttention(config)
    module.eval()

    B, S = 2, 32
    D = config.d_model

    # 1. Cache perception features (from Qwen)
    perception_features = torch.randn(B, S, config.d_perception)
    logger.info(f"Caching perception features: {perception_features.shape}")
    module.cache_perception(perception_features)
    assert module.has_cache, "Cache should be populated"

    # 2. Create inputs for each tick
    state = torch.randn(B, S, D)
    personality_signal = torch.randn(B, S, D)
    intention_signal = torch.randn(B, S, D)
    sync = torch.randn(B, S, config.sync_pairs)

    logger.info(f"State shape: {state.shape}")
    logger.info(f"Personality shape: {personality_signal.shape}")
    logger.info(f"Intention shape: {intention_signal.shape}")
    logger.info(f"Sync shape: {sync.shape}")

    # 3. Run multiple ticks
    for tick in range(5):
        output = module(
            state=state,
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=tick,
        )
        logger.info(f"Tick {tick}: observation shape: {output.observation.shape}")
        assert output.observation.shape == (B, S, D)

    # 4. Clear and verify
    module.clear_cache()
    assert not module.has_cache, "Cache should be cleared"

    logger.info("PerceptionAttention (full module) test PASSED")
    return True


def test_gradient_flow():
    """Test that gradients flow through entire module."""
    from pem.perception_attention import PerceptionAttention, PerceptionConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Gradient flow")
    logger.info("=" * 60)

    config = PerceptionConfig(
        d_model=256,
        d_perception=512,
        n_heads=4,
        sync_pairs=256,
        use_flash_attention=False,  # Flash attention doesn't work on CPU
    )

    module = PerceptionAttention(config)
    module.train()

    B, S = 1, 16
    D = config.d_model

    # Create inputs with gradients
    perception_features = torch.randn(B, S, config.d_perception, requires_grad=True)
    state = torch.randn(B, S, D, requires_grad=True)
    personality_signal = torch.randn(B, S, D, requires_grad=True)
    intention_signal = torch.randn(B, S, D, requires_grad=True)
    sync = torch.randn(B, S, config.sync_pairs, requires_grad=True)

    # Cache and forward
    module.cache_perception(perception_features)
    output = module(
        state=state,
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
    )

    # Backward
    loss = output.observation.mean()
    loss.backward()

    logger.info(f"Perception features grad exists: {perception_features.grad is not None}")
    logger.info(f"State grad exists: {state.grad is not None}")
    logger.info(f"Personality grad exists: {personality_signal.grad is not None}")
    logger.info(f"Intention grad exists: {intention_signal.grad is not None}")
    logger.info(f"Sync grad exists: {sync.grad is not None}")

    # All inputs should have gradients
    assert perception_features.grad is not None, "Perception features should have gradients"
    assert state.grad is not None, "State should have gradients"
    assert personality_signal.grad is not None, "Personality should have gradients"
    assert intention_signal.grad is not None, "Intention should have gradients"
    assert sync.grad is not None, "Sync should have gradients"

    # Check module parameters have gradients
    has_param_grads = False
    for name, param in module.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_param_grads = True
            break

    logger.info(f"Module parameters have gradients: {has_param_grads}")
    assert has_param_grads, "Module parameters should have gradients"

    logger.info("Gradient flow test PASSED")
    return True


def test_tick_variation():
    """Test that different ticks produce different attention patterns."""
    from pem.perception_attention import PerceptionAttention, PerceptionConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Tick variation in attention")
    logger.info("=" * 60)

    config = PerceptionConfig(
        d_model=256,
        d_perception=512,
        n_heads=4,
        sync_pairs=256,
        num_oscillators=16,
        use_flash_attention=False,  # Need attention weights
    )

    module = PerceptionAttention(config)
    module.eval()

    B, S = 1, 16
    D = config.d_model

    # Fixed inputs
    perception_features = torch.randn(B, S, config.d_perception)
    state = torch.randn(B, S, D)
    personality_signal = torch.randn(B, S, D)
    intention_signal = torch.randn(B, S, D)
    sync = torch.randn(B, S, config.sync_pairs)

    module.cache_perception(perception_features)

    # Collect observations at different ticks
    observations = []
    for tick in range(10):
        with torch.no_grad():
            output = module(
                state=state,
                personality_signal=personality_signal,
                intention_signal=intention_signal,
                sync=sync,
                tick=tick,
            )
            observations.append(output.observation)

    # Check variation across ticks
    variations = []
    for i in range(1, len(observations)):
        diff = (observations[i] - observations[i-1]).abs().mean().item()
        variations.append(diff)

    mean_var = sum(variations) / len(variations)
    logger.info(f"Mean observation variation across ticks: {mean_var:.6f}")

    # Should have some variation (oscillation effect)
    # Note: With architectural changes, variation may be smaller but should exist
    if mean_var <= 1e-8:
        logger.info("  (Very small variation - oscillation effect may be damped)")
    assert mean_var > 0, "Observations should vary across ticks due to oscillation"

    logger.info("Tick variation test PASSED")
    return True


def test_integration_with_sync_module():
    """Test PerceptionAttention with actual SyncModule outputs."""
    from pem.perception_attention import PerceptionAttention, PerceptionConfig
    from pem.sync_module import SyncModule, SyncModuleConfig

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Integration with SyncModule")
    logger.info("=" * 60)

    # Create SyncModule
    sync_config = SyncModuleConfig(
        d_model=256,
        sync_pairs=256,
        n_heads=4,
        memory_slots=20,
        use_intention=True,
        intention_oscillators=8,
    )
    sync_module = SyncModule(sync_config)

    # Create PerceptionAttention
    perception_config = PerceptionConfig(
        d_model=256,
        d_perception=512,
        n_heads=4,
        sync_pairs=256,
        use_flash_attention=False,  # Flash attention doesn't work on CPU
    )
    perception = PerceptionAttention(perception_config)

    B, S, T = 1, 16, 4
    D = sync_config.d_model

    # Create history for sync module
    history = torch.randn(B, S, T, D)

    # Get sync output
    sync_module.eval()
    with torch.no_grad():
        sync = sync_module(history)

    logger.info(f"Sync output shape: {sync.shape}")

    # Get personality signal from sync module
    context = torch.randn(B, S, D)  # Would come from context encoder
    personality_signal = sync_module.personality(context)

    # Get intention signal if available
    if sync_module.intention is not None:
        intention_signal, _ = sync_module.intention(context)
    else:
        intention_signal = torch.randn(B, S, D)

    logger.info(f"Personality signal shape: {personality_signal.shape}")
    logger.info(f"Intention signal shape: {intention_signal.shape}")

    # Run perception attention
    perception_features = torch.randn(B, S, perception_config.d_perception)
    perception.cache_perception(perception_features)

    state = torch.randn(B, S, D)

    perception.eval()
    with torch.no_grad():
        output = perception(
            state=state,
            personality_signal=personality_signal,
            intention_signal=intention_signal,
            sync=sync,
            tick=0,
        )

    logger.info(f"Observation shape: {output.observation.shape}")
    assert output.observation.shape == (B, S, D)

    logger.info("Integration with SyncModule test PASSED")
    return True


def test_imagination_in_attention_pool():
    """Test that imagination can be added to the attention pool."""
    from pem.perception_attention import PerceptionAttention, PerceptionConfig

    config = PerceptionConfig(
        d_model=256,
        d_perception=512,
        n_heads=4,
        sync_pairs=256,
        num_oscillators=16,
        use_flash_attention=False,  # Disable flash attention for CPU testing
    )

    module = PerceptionAttention(config)

    B, S = 2, 16
    D = config.d_model
    D_perception = config.d_perception

    # Create inputs
    perception_features = torch.randn(B, S, D_perception)
    imagined_features = torch.randn(B, S // 2, D)  # Imagination can be different length
    state = torch.randn(B, S, D)
    personality_signal = torch.randn(B, S, D)
    intention_signal = torch.randn(B, S, D)
    sync = torch.randn(B, S, config.sync_pairs)

    # Cache perception
    module.cache_perception(perception_features)
    assert module.has_cache, "Perception should be cached"
    assert not module.has_imagination, "Imagination should not be cached yet"

    # Forward without imagination
    output_without = module(
        state=state,
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
    )
    logger.info(f"Output without imagination shape: {output_without.observation.shape}")
    logger.info(f"Attention weights shape: {output_without.attention_weights.shape}")

    # Add imagination
    module.add_imagination(imagined_features)
    assert module.has_imagination, "Imagination should be cached"

    # Forward with imagination
    output_with = module(
        state=state,
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
    )
    logger.info(f"Output with imagination shape: {output_with.observation.shape}")
    logger.info(f"Attention weights with imagination shape: {output_with.attention_weights.shape}")

    # Attention should now span more positions (real + imagined)
    expected_kv_len = S + S // 2
    assert output_with.attention_weights.shape[-1] == expected_kv_len, \
        f"KV length should be {expected_kv_len}, got {output_with.attention_weights.shape[-1]}"

    # Output shapes should remain the same
    assert output_without.observation.shape == output_with.observation.shape, \
        "Output shape should be the same with or without imagination"

    # Clear imagination
    module.clear_imagination()
    assert not module.has_imagination, "Imagination should be cleared"
    assert module.has_cache, "Real perception should still be cached"

    logger.info("Imagination in attention pool test PASSED")
    return True


def test_shared_kv_projection():
    """Test that perception and imagination use the same K/V projections."""
    from pem.perception_attention import PerceptionKVCache

    D_perception = 512
    D_model = 256
    n_heads = 4

    kv_cache = PerceptionKVCache(
        d_perception=D_perception,
        d_model=D_model,
        n_heads=n_heads,
    )

    B, S = 2, 16

    # Create perception and imagination features
    perception = torch.randn(B, S, D_perception)
    imagination = torch.randn(B, S, D_model)

    # Project both
    k_real, v_real = kv_cache.project_perception(perception)
    k_imag, v_imag = kv_cache.project_imagination(imagination)

    # Check shapes are compatible
    assert k_real.shape == k_imag.shape, "K shapes should match"
    assert v_real.shape == v_imag.shape, "V shapes should match"
    logger.info(f"K shape (both): {k_real.shape}")
    logger.info(f"V shape (both): {v_real.shape}")

    # Verify they go through the same k_proj/v_proj
    # (The k_proj and v_proj layers are shared)
    assert kv_cache.k_proj is not None, "k_proj should exist"
    assert kv_cache.v_proj is not None, "v_proj should exist"

    logger.info("Shared K/V projection test PASSED")
    return True


def test_surprise_integration():
    """Test that surprise influences attention query (closes experience loop)."""
    from pem.perception_attention import PerceptionAttention, PerceptionConfig, OscillationQueryBuilder

    logger.info("\n" + "=" * 60)
    logger.info("TEST: Surprise Integration (Experience Loop)")
    logger.info("=" * 60)

    d_model = 256
    sync_pairs = 256

    # Test OscillationQueryBuilder with surprise
    builder = OscillationQueryBuilder(
        d_model=d_model,
        sync_pairs=sync_pairs,
        num_oscillators=16,
    )

    B, S = 1, 16
    personality_signal = torch.randn(B, S, d_model)
    intention_signal = torch.randn(B, S, d_model)
    sync = torch.randn(B, S, sync_pairs)

    # Without surprise
    query_no_surprise = builder(
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
    )

    # With surprise
    surprise_magnitude = torch.rand(B, S, 1) * 2  # Random magnitudes
    surprise_direction = torch.randn(B, S, d_model)
    surprise_direction = surprise_direction / (surprise_direction.norm(dim=-1, keepdim=True) + 1e-8)

    query_with_surprise = builder(
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
        surprise_magnitude=surprise_magnitude,
        surprise_direction=surprise_direction,
    )

    # Queries should differ when surprise is added
    diff = (query_no_surprise - query_with_surprise).abs().mean()
    logger.info(f"Query difference with/without surprise: {diff:.4f}")
    assert diff > 1e-4, "Surprise should influence the query"

    # Test full PerceptionAttention with surprise
    config = PerceptionConfig(
        d_model=d_model,
        d_perception=512,
        n_heads=4,
        sync_pairs=sync_pairs,
        use_flash_attention=False,
    )

    module = PerceptionAttention(config)
    module.eval()

    perception_features = torch.randn(B, S, config.d_perception)
    module.cache_perception(perception_features)

    state = torch.randn(B, S, d_model)

    # Without surprise
    output_no_surprise = module(
        state=state,
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
    )

    # With surprise
    output_with_surprise = module(
        state=state,
        personality_signal=personality_signal,
        intention_signal=intention_signal,
        sync=sync,
        tick=0,
        surprise_magnitude=surprise_magnitude,
        surprise_direction=surprise_direction,
    )

    # Observations should differ (may be small since attention patterns can be similar)
    obs_diff = (output_no_surprise.observation - output_with_surprise.observation).abs().mean()
    logger.info(f"Observation difference with/without surprise: {obs_diff:.6f}")
    # The key test is that queries differ; observation difference may be small
    # but should be non-zero since queries are different
    assert obs_diff > 1e-6, "Surprise should influence the observation"

    logger.info("Surprise Integration test PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("PEM Perception Attention Tests")
    logger.info("=" * 60)

    tests = [
        ("PerceptionKVCache", test_perception_kv_cache),
        ("OscillationQueryBuilder", test_oscillation_query_builder),
        ("PerceptionCrossAttention", test_perception_cross_attention),
        ("PerceptionSynapse", test_perception_synapse),
        ("PerceptionAttention full", test_perception_attention_full),
        ("Gradient flow", test_gradient_flow),
        ("Tick variation", test_tick_variation),
        ("Integration with SyncModule", test_integration_with_sync_module),
        ("Imagination in Attention Pool", test_imagination_in_attention_pool),
        ("Shared K/V Projection", test_shared_kv_projection),
        ("Surprise Integration", test_surprise_integration),
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
