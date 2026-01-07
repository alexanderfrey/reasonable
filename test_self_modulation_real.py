"""
Test self-modulation and certainty calibration on real training data.

Analyzes how the confidence gate varies with meta-surprise and
whether the system learns to modulate its outputs appropriately.
Also tests the new certainty calibration system (Phase 1-2).

Usage:
    python test_self_modulation_real.py --max_steps 1000
"""

import argparse
import json
import logging
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiential import (
    ExperientialStream,
    combined_experiential_loss,
)
from certainty import expected_calibration_error
from train_experiential import (
    HiddenStateExtractor,
    load_model,
    DEFAULT_CHECKPOINT,
    DEFAULT_DATA_DIR,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_with_self_modulation(
    extractor: HiddenStateExtractor,
    experiential: ExperientialStream,
    dataloader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_steps: int = 1000,
    log_interval: int = 50,
    meta_weight: float = 0.5,
    self_mod_weight: float = 0.1,
    certainty_weight: float = 0.1,
):
    """
    Train experiential module tracking self-modulation and certainty metrics.
    """
    extractor.model.eval()
    for param in extractor.model.parameters():
        param.requires_grad = False

    experiential.train()

    history = {
        'exp_loss': [],
        'meta_loss': [],
        'self_mod_loss': [],
        'certainty_loss': [],
        'meta_surprise': [],
        'confidence_mean': [],
        'confidence_std': [],
        'modulation_magnitude': [],  # How much modulated_output differs from h_end
        'surprise': [],
        'certainty': [],
        'accuracy': [],  # For ECE computation
    }

    # For ECE computation at end
    all_certainties = []
    all_was_correct = []

    start_time = time.time()
    step = 0

    pbar = tqdm(total=max_steps, desc="Training")

    for batch in dataloader:
        if step >= max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        batch_size = input_ids.size(0)
        seq_len = input_ids.size(1)

        # Reset state for each batch
        experiential.reset_state(batch_size=batch_size)

        # Get hidden states from frozen backbone
        with torch.no_grad():
            _, hidden_states = extractor(input_ids)

            # Compute logits to determine was_correct
            # Use lm_head on last position hidden state
            last_hidden = hidden_states[:, -1, :]  # [B, d_model]
            logits = extractor.model.lm_head(last_hidden)  # [B, vocab]
            predictions = logits.argmax(dim=-1)  # [B]

            # Target is what would come after the sequence
            # Since we don't have it, use a proxy: check if top prediction
            # matches the actual last token (shifted by 1)
            # Or use top-k accuracy as soft correctness
            targets = input_ids[:, -1]  # Last token as proxy target
            was_correct = (predictions == targets).float()  # [B]

        # Forward through experiential module
        exp_output = experiential(hidden_states)

        # Combined loss (includes self-modulation and certainty calibration loss)
        loss, loss_dict = combined_experiential_loss(
            exp_output,
            exp_weight=1.0,
            meta_weight=meta_weight,
            self_mod_weight=self_mod_weight,
            certainty_calibration_weight=certainty_weight,
            was_correct=was_correct,
        )

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(experiential.parameters(), max_norm=1.0)
        optimizer.step()

        # Track self-modulation and certainty metrics
        with torch.no_grad():
            conf_gate = exp_output['confidence_gate']  # [B, d_model]
            modulated = exp_output['modulated_output']  # [B, d_model]
            h_end = exp_output['h_end']  # [B, d_model] - raw world state

            # Mean confidence across dimensions
            conf_mean = conf_gate.mean().item()
            conf_std = conf_gate.std().item()

            # How much did modulation change the output?
            # Compare modulated_output to h_end (raw world state)
            # This shows how much self-knowledge adjusted the committed output
            mod_diff = (modulated - h_end).norm(dim=-1) / (h_end.norm(dim=-1) + 1e-8)
            mod_magnitude = mod_diff.mean().item()

            # Certainty metrics
            certainty = exp_output.get('certainty')
            if certainty is not None:
                certainty_mean = certainty.mean().item()
                # Collect for ECE computation
                all_certainties.append(certainty.detach().cpu())
                all_was_correct.append(was_correct.detach().cpu())
            else:
                certainty_mean = 0.5

        history['exp_loss'].append(loss_dict['exp_loss'])
        history['meta_loss'].append(loss_dict.get('meta_loss', 0))
        history['self_mod_loss'].append(loss_dict.get('self_mod_loss', 0))
        history['certainty_loss'].append(loss_dict.get('certainty_calibration_loss', 0))
        history['meta_surprise'].append(loss_dict.get('mean_meta_surprise', 0))
        history['confidence_mean'].append(loss_dict.get('mean_confidence', conf_mean))
        history['confidence_std'].append(conf_std)
        history['modulation_magnitude'].append(mod_magnitude)
        history['surprise'].append(exp_output['surprise'].mean().item())
        history['certainty'].append(certainty_mean)
        history['accuracy'].append(was_correct.mean().item())

        # Log
        if step % log_interval == 0:
            pbar.set_postfix({
                'ms': f"{loss_dict.get('mean_meta_surprise', 0):.3f}",
                'conf': f"{loss_dict.get('mean_confidence', conf_mean):.3f}",
                'cert': f"{certainty_mean:.3f}",
                'cert_loss': f"{loss_dict.get('certainty_calibration_loss', 0):.4f}",
            })

        pbar.update(1)
        step += 1

    pbar.close()
    elapsed = time.time() - start_time
    logger.info(f"Training complete: {step} steps in {elapsed:.1f}s")

    # Compute final ECE
    ece_value = None
    ece_details = None
    if all_certainties:
        all_cert = torch.cat(all_certainties)
        all_corr = torch.cat(all_was_correct)
        ece_value, ece_details = expected_calibration_error(all_cert, all_corr)
        logger.info(f"Final ECE: {ece_value.item():.4f}")

    return history, ece_value, ece_details


def analyze_self_modulation(history: dict):
    """Analyze self-modulation patterns."""
    n = len(history['meta_surprise'])

    # Split into phases
    phase_size = n // 4
    phases = {
        'early': (0, phase_size),
        'mid-early': (phase_size, 2 * phase_size),
        'mid-late': (2 * phase_size, 3 * phase_size),
        'late': (3 * phase_size, n),
    }

    print("\n" + "=" * 80)
    print("SELF-MODULATION ANALYSIS")
    print("=" * 80)

    print(f"\n{'Phase':<12} {'Meta-Surp':>12} {'Confidence':>12} {'SM Loss':>12} {'Modulation':>12}")
    print("-" * 80)

    phase_data = {}
    for phase_name, (start, end) in phases.items():
        ms = sum(history['meta_surprise'][start:end]) / (end - start)
        conf = sum(history['confidence_mean'][start:end]) / (end - start)
        sm_loss = sum(history['self_mod_loss'][start:end]) / (end - start)
        mod = sum(history['modulation_magnitude'][start:end]) / (end - start)

        phase_data[phase_name] = {'ms': ms, 'conf': conf, 'sm_loss': sm_loss, 'mod': mod}
        print(f"{phase_name:<12} {ms:>12.4f} {conf:>12.4f} {sm_loss:>12.4f} {mod:>12.4f}")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    # Compute correlation between meta-surprise and confidence
    ms_tensor = torch.tensor(history['meta_surprise'])
    conf_tensor = torch.tensor(history['confidence_mean'])
    mod_tensor = torch.tensor(history['modulation_magnitude'])

    # Pearson correlation
    ms_centered = ms_tensor - ms_tensor.mean()
    conf_centered = conf_tensor - conf_tensor.mean()
    mod_centered = mod_tensor - mod_tensor.mean()

    ms_conf_corr = (ms_centered * conf_centered).sum() / (ms_centered.norm() * conf_centered.norm() + 1e-8)
    ms_mod_corr = (ms_centered * mod_centered).sum() / (ms_centered.norm() * mod_centered.norm() + 1e-8)

    print(f"\nCorrelation(meta-surprise, confidence): {ms_conf_corr:.4f}")
    print(f"Correlation(meta-surprise, modulation): {ms_mod_corr:.4f}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    early_ms = phase_data['early']['ms']
    late_ms = phase_data['late']['ms']
    early_conf = phase_data['early']['conf']
    late_conf = phase_data['late']['conf']
    early_mod = phase_data['early']['mod']
    late_mod = phase_data['late']['mod']

    if late_ms < early_ms:
        reduction = (early_ms - late_ms) / early_ms * 100
        print(f"\n✓ Meta-surprise DECREASED by {reduction:.1f}%")
        print("  → System learned to predict its own surprise")
    else:
        print("\n✗ Meta-surprise did not decrease")

    if late_conf > early_conf:
        increase = (late_conf - early_conf) / early_conf * 100
        print(f"\n✓ Confidence INCREASED by {increase:.1f}%")
        print("  → System became more confident as it learned itself")
    elif late_conf < early_conf:
        decrease = (early_conf - late_conf) / early_conf * 100
        print(f"\n~ Confidence decreased by {decrease:.1f}%")
        print("  → System became more conservative (may need tuning)")

    if ms_conf_corr < -0.1:
        print(f"\n✓ Negative correlation between meta-surprise and confidence")
        print("  → High uncertainty → lower confidence (expected behavior)")
    elif ms_conf_corr > 0.1:
        print(f"\n? Positive correlation between meta-surprise and confidence")
        print("  → May need more training for self-modulation to emerge")
    else:
        print(f"\n~ Weak correlation between meta-surprise and confidence")
        print("  → Self-modulation learning may need more steps")

    if late_mod < early_mod:
        decrease = (early_mod - late_mod) / early_mod * 100
        print(f"\n✓ Modulation magnitude DECREASED by {decrease:.1f}%")
        print("  → System needs less correction as it learns")

    print("\n" + "=" * 70)

    return {
        'early_ms': early_ms,
        'late_ms': late_ms,
        'early_conf': early_conf,
        'late_conf': late_conf,
        'ms_conf_corr': ms_conf_corr.item(),
        'ms_mod_corr': ms_mod_corr.item(),
    }


def analyze_certainty_calibration(history: dict, ece_value, ece_details):
    """Analyze certainty calibration patterns."""
    print("\n" + "=" * 80)
    print("CERTAINTY CALIBRATION ANALYSIS")
    print("=" * 80)

    n = len(history.get('certainty', []))
    if n == 0:
        print("\nNo certainty data recorded.")
        return {}

    # Split into phases
    phase_size = n // 4
    phases = {
        'early': (0, phase_size),
        'mid-early': (phase_size, 2 * phase_size),
        'mid-late': (2 * phase_size, 3 * phase_size),
        'late': (3 * phase_size, n),
    }

    print(f"\n{'Phase':<12} {'Certainty':>12} {'Accuracy':>12} {'Cert Loss':>12} {'Gap':>12}")
    print("-" * 80)

    phase_data = {}
    for phase_name, (start, end) in phases.items():
        cert = sum(history['certainty'][start:end]) / (end - start)
        acc = sum(history['accuracy'][start:end]) / (end - start)
        cert_loss = sum(history['certainty_loss'][start:end]) / (end - start)
        gap = abs(cert - acc)  # Calibration gap

        phase_data[phase_name] = {'cert': cert, 'acc': acc, 'cert_loss': cert_loss, 'gap': gap}
        print(f"{phase_name:<12} {cert:>12.4f} {acc:>12.4f} {cert_loss:>12.4f} {gap:>12.4f}")

    # Correlation analysis
    print("\n" + "=" * 70)
    print("CERTAINTY-ACCURACY CORRELATION")
    print("=" * 70)

    cert_tensor = torch.tensor(history['certainty'])
    acc_tensor = torch.tensor(history['accuracy'])
    surp_tensor = torch.tensor(history['surprise'])

    cert_centered = cert_tensor - cert_tensor.mean()
    acc_centered = acc_tensor - acc_tensor.mean()
    surp_centered = surp_tensor - surp_tensor.mean()

    cert_acc_corr = (cert_centered * acc_centered).sum() / (cert_centered.norm() * acc_centered.norm() + 1e-8)
    cert_surp_corr = (cert_centered * surp_centered).sum() / (cert_centered.norm() * surp_centered.norm() + 1e-8)

    print(f"\nCorrelation(certainty, accuracy): {cert_acc_corr:.4f}")
    print(f"Correlation(certainty, surprise): {cert_surp_corr:.4f}")

    # ECE analysis
    if ece_value is not None:
        print(f"\nExpected Calibration Error (ECE): {ece_value.item():.4f}")
        if ece_details is not None:
            print("\nPer-bin calibration:")
            bin_acc = ece_details['bin_accuracies']
            bin_conf = ece_details['bin_confidences']
            bin_counts = ece_details['bin_counts']
            for i, (acc, conf, count) in enumerate(zip(bin_acc, bin_conf, bin_counts)):
                if count > 0:
                    print(f"  Bin {i}: conf={conf:.3f}, acc={acc:.3f}, gap={abs(conf-acc):.3f}, n={count}")

    # Interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    early_cert = phase_data['early']['cert']
    late_cert = phase_data['late']['cert']
    early_gap = phase_data['early']['gap']
    late_gap = phase_data['late']['gap']

    if late_gap < early_gap:
        reduction = (early_gap - late_gap) / (early_gap + 1e-8) * 100
        print(f"\n✓ Calibration gap DECREASED by {reduction:.1f}%")
        print("  → Certainty is becoming better calibrated to accuracy")
    else:
        print("\n✗ Calibration gap did not decrease")

    if cert_acc_corr > 0.3:
        print(f"\n✓ Strong positive correlation between certainty and accuracy")
        print("  → Model is more certain when it's more likely to be correct")
    elif cert_acc_corr > 0:
        print(f"\n~ Weak positive correlation between certainty and accuracy")
        print("  → Certainty calibration is emerging")
    else:
        print(f"\n✗ Negative/zero correlation between certainty and accuracy")
        print("  → Certainty needs more training")

    if cert_surp_corr < -0.3:
        print(f"\n✓ Strong negative correlation between certainty and surprise")
        print("  → Low surprise → high certainty (expected behavior)")

    if ece_value is not None:
        if ece_value.item() < 0.1:
            print(f"\n✓ ECE is low ({ece_value.item():.4f}) - well calibrated")
        elif ece_value.item() < 0.2:
            print(f"\n~ ECE is moderate ({ece_value.item():.4f}) - reasonably calibrated")
        else:
            print(f"\n✗ ECE is high ({ece_value.item():.4f}) - needs more calibration training")

    print("\n" + "=" * 70)

    return {
        'early_cert': early_cert,
        'late_cert': late_cert,
        'early_gap': early_gap,
        'late_gap': late_gap,
        'cert_acc_corr': cert_acc_corr.item(),
        'cert_surp_corr': cert_surp_corr.item(),
        'ece': ece_value.item() if ece_value is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Test self-modulation on real data")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--meta_weight", type=float, default=0.5)
    parser.add_argument("--self_mod_weight", type=float, default=0.1,
                        help="Weight for self-modulation loss")
    parser.add_argument("--certainty_weight", type=float, default=0.1,
                        help="Weight for certainty calibration loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    extractor = HiddenStateExtractor(model)

    # Create experiential module with meta-surprise AND self-modulation
    experiential = ExperientialStream(
        d_model=config.d_model,
        use_meta_surprise=True,  # Enables self-modulation
        use_affect=True,
        use_persistent_state=True
    ).to(device)

    n_params = sum(p.numel() for p in experiential.parameters())
    logger.info(f"ExperientialStream: {n_params:,} parameters")

    # Load data
    from pretrain import PretokenizedDataset
    import glob

    meta_files = glob.glob(os.path.join(args.data_dir, "training*_metadata.json"))
    if not meta_files:
        raise FileNotFoundError(f"No training metadata found in {args.data_dir}")

    with open(meta_files[0]) as f:
        meta = json.load(f)

    token_file = meta.get('token_file') or meta_files[0].replace('_metadata.json', '_tokens.bin')
    num_examples = min(meta.get('num_examples', 10000), args.max_steps * args.batch_size * 2)

    logger.info(f"Loading data from {token_file}")
    dataset = PretokenizedDataset(
        token_file, num_examples, args.seq_len, args.seq_len, data_type="Train"
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    logger.info(f"Loaded {len(dataset)} examples")

    # Optimizer
    optimizer = torch.optim.AdamW(
        experiential.parameters(),
        lr=args.lr,
        weight_decay=0.01
    )

    # Train
    logger.info(f"\nTraining with self-modulation (weight={args.self_mod_weight}) and certainty (weight={args.certainty_weight})...")
    history, ece_value, ece_details = train_with_self_modulation(
        extractor=extractor,
        experiential=experiential,
        dataloader=dataloader,
        optimizer=optimizer,
        device=device,
        max_steps=args.max_steps,
        log_interval=50,
        meta_weight=args.meta_weight,
        self_mod_weight=args.self_mod_weight,
        certainty_weight=args.certainty_weight,
    )

    # Analyze self-modulation
    sm_results = analyze_self_modulation(history)

    # Analyze certainty calibration
    cert_results = analyze_certainty_calibration(history, ece_value, ece_details)

    # Save
    output_path = "self_modulation_results.pt"
    torch.save({
        'history': history,
        'self_modulation_results': sm_results,
        'certainty_results': cert_results,
        'ece_value': ece_value.item() if ece_value is not None else None,
        'ece_details': {k: v.cpu() for k, v in ece_details.items()} if ece_details else None,
        'args': vars(args),
    }, output_path)
    logger.info(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
