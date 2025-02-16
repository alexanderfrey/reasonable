import torch
from colorama import init, Fore, Style
import argparse
from typing import List, Tuple, Optional, Any
from torch.nn import functional as F
import tiktoken

# Import your model architecture
from model import GPTConfig, MoEGPT, GPT  # Assuming model.py contains your model code


class AlpacaFormatter:
    def __init__(
        self,
        instruction_token: str = "[INST]",
        response_token: str = "[/INST]",
        end_token: str = "</s>",
        max_length: int = 1024,
        pad_token_id: int = 50256,
    ):
        self.instruction_token = instruction_token
        self.response_token = response_token
        self.end_token = end_token
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def format_prompt(
        self,
        instruction: str,
        response: str = "",
        input_context: str = None,
        system: str = None,
    ) -> str:
        """Format using Alpaca style with optional system prompt"""
        formatted = f"{self.instruction_token}"

        # Add system prompt if provided
        if system:
            formatted += f"### System:\n{system}\n\n"

        formatted += f"### Instruction:\n{instruction}\n"

        if input_context:
            formatted += f"\n### Input:\n{input_context}\n"

        if response:
            formatted += f"\n### Response:\n{response}"
        else:
            formatted += f"\n### Response:\n"

        formatted += f"{self.response_token}{self.end_token}"
        return formatted


def get_config_from_checkpoint(checkpoint_path: str) -> Tuple[GPTConfig, int]:
    """
    Infer config parameters from checkpoint state dict, including auxiliary heads configuration.

    Args:
        checkpoint_path: Path to the model checkpoint

    Returns:
        tuple: (GPTConfig, number of experts)
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    # Count number of blocks
    n_layer = (
        max(
            [int(k.split(".")[1]) for k in state_dict.keys() if k.startswith("blocks.")]
        )
        + 1
    )

    # Check if bias is used
    has_bias = any(["bias" in k for k in state_dict.keys()])

    # Get accurate dimensions from latent queries
    latent_queries = state_dict["blocks.0.attn.latent_queries"]
    n_latents = latent_queries.shape[1]  # 64
    n_head = latent_queries.shape[2]  # 6
    head_dim = latent_queries.shape[3]  # 128
    n_embd = n_head * head_dim  # 768

    # Get vocab size from token embedding
    vocab_size = state_dict["token_embedding.weight"].shape[0]

    # Count number of auxiliary heads
    aux_head_keys = [k for k in state_dict.keys() if k.startswith("aux_heads.")]
    n_aux_heads = len(
        set(
            [
                int(k.split(".")[1])
                for k in aux_head_keys
                if k.split(".")[-1] == "weight"
            ]
        )
    )

    # Determine number of experts
    expert_keys = [k for k in state_dict.keys() if "experts" in k]
    n_experts = (
        len(set([int(k.split("experts.")[1].split(".")[0]) for k in expert_keys]))
        if expert_keys
        else 0
    )

    # Create config with auxiliary heads
    config = GPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        n_aux_heads=n_aux_heads,  # Add number of auxiliary heads
        bias=has_bias,
        block_size=1024,  # You might want to make this configurable
    )

    return config, n_experts


def load_model(checkpoint_path: str, is_moe: bool = False) -> MoEGPT:
    """Load model from checkpoint with inferred config."""
    config, n_experts = get_config_from_checkpoint(checkpoint_path)
    print(f"\nInferred model configuration:")
    print(f"Number of layers: {config.n_layer}")
    print(f"Number of heads: {config.n_head}")
    print(f"Embedding dimension: {config.n_embd}")
    print(f"Vocabulary size: {config.vocab_size}")
    print(f"Using bias: {config.bias}")
    print(f"Number of auxiliary heads: {config.n_aux_heads}")
    print(f"Number of experts per MoE layer: {n_experts}")

    if is_moe:
        model = MoEGPT(config, num_experts=n_experts if n_experts > 0 else 8)
    else:
        model = GPT(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    # Handle missing auxiliary heads in the state dict
    if config.n_aux_heads > 0:
        aux_head_keys = [k for k in state_dict.keys() if k.startswith("aux_heads.")]
        if not aux_head_keys:
            print(
                "Warning: Auxiliary heads specified in config but not found in checkpoint"
            )

    # Load state dict
    model.load_state_dict(state_dict)
    return model


def get_next_token_probs(
    model: GPT,
    input_ids: torch.Tensor,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
) -> torch.Tensor:
    """Get probability distribution over next tokens."""
    with torch.no_grad():
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        # Filter by top-k if specified
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        # Filter by top-p if specified
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return probs


def generate_sequence(
    model: Any,
    tokenizer,
    prompt: str,
    formatter: Optional[AlpacaFormatter] = None,
    instruction_mode: bool = False,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> str:
    """Generate text sequence from prompt with optional instruction formatting."""
    model.to(device)
    model.eval()

    # Apply instruction formatting if requested
    if instruction_mode and formatter:
        # Default simple instruction task if none provided
        if "### Instruction:" not in prompt:
            prompt = formatter.format_prompt(
                instruction="""As a member of an international diplomatic think tank, you are tasked with advising the leaders of NATO on the strategic direction for Ukraine amidst growing tensions with Russia. The recent visit by U.S. Defense Secretary Pete Hegseth to NATO has prompted a reevaluation of Ukraine's future. You must navigate three key decision points: NATO membership for Ukraine, the potential for a negotiated settlement, and the structure of a peacekeeping force. Consider the interests and constraints of stakeholders including Ukraine, Russia, NATO members, and the United States.""",
                input_context="""context": "After Russia's 2022 invasion of Ukraine, NATO and its allies have been supporting Ukraine with over $126 billion in military assistance. However, the U.S. now suggests that NATO membership for Ukraine is unrealistic and advocates for a negotiated settlement with Russia. Defense Secretary Hegseth proposes a peacekeeping force without U.S. troops and without Article Five protections. The U.K. recently chaired the Ukraine Defense Contact Group, a role previously held by the U.S.""",
            )

    # Tokenize prompt with tiktoken
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)  # [1, seq_len]

    # Generate tokens
    generated_tokens = []
    for _ in range(max_new_tokens):
        # Get next token probabilities
        next_token_probs = get_next_token_probs(
            model, input_ids, temperature, top_k, top_p
        )

        # Sample next token and reshape to match input_ids dimensions [1, 1]
        next_token = torch.multinomial(next_token_probs, num_samples=1).view(1, 1)

        # Check for end token if using instruction format
        if (
            instruction_mode
            and formatter
            and next_token.item() == tokenizer.encode(formatter.end_token)[0]
        ):
            break

        # Append to generated sequence
        generated_tokens.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

    # Decode generated tokens with tiktoken
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def analyze_expert_usage_detailed(
    model: MoEGPT,
    input_text: str,
    tokenizer,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> dict:
    """Analyze expert utilization with detailed metrics including gating weights and load balancing.

    Returns:
        dict: Dictionary containing detailed statistics for each MoE layer
    """
    model.to(device)
    model.eval()

    # Prepare input
    tokens = tokenizer.encode(input_text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    seq_len = input_ids.size(1)

    layer_stats = {}

    with torch.no_grad():
        # Initial embeddings
        x = model.token_embedding(input_ids) + model.position_embedding(
            torch.arange(input_ids.size(1), device=device)
        )

        # Process each layer
        for i, block in enumerate(model.blocks):
            if hasattr(block, "moe"):
                # Get layer normalized input for router
                normalized_input = block.ln2(x)

                # Get router logits and gate values
                router_logits = block.moe.router(normalized_input)
                gate_values = torch.softmax(router_logits, dim=-1)

                # Get top-k experts and their gate values
                top_gates, top_indices = torch.topk(
                    gate_values, k=block.moe.top_k, dim=-1
                )

                # Initialize layer statistics
                layer_stats[i] = {
                    "num_experts": block.moe.num_experts,
                    "top_k": block.moe.top_k,
                    "seq_length": seq_len,
                    "expert_stats": {},
                    "load_balancing": {
                        "capacity_fulfillment": torch.zeros(
                            block.moe.num_experts, device=device
                        ),
                        "unused_expert_count": 0,
                        "overflow_expert_count": 0,
                    },
                }

                # Calculate ideal balanced load per expert
                # Each token activates top_k experts, so total activations = seq_len * top_k
                ideal_load_per_expert = (
                    seq_len * block.moe.top_k
                ) / block.moe.num_experts

                # Analyze each expert
                for expert_idx in range(block.moe.num_experts):
                    # Get mask for where this expert was selected
                    expert_mask = top_indices == expert_idx

                    # Get gate values when this expert was selected
                    expert_gates = torch.where(
                        expert_mask, top_gates, torch.zeros_like(top_gates)
                    )

                    # Calculate statistics
                    total_selections = expert_mask.sum().item()
                    avg_gate_value = (
                        expert_gates.sum().item() / total_selections
                        if total_selections > 0
                        else 0
                    )

                    # Calculate load metrics
                    capacity_fulfillment = total_selections / ideal_load_per_expert

                    layer_stats[i]["expert_stats"][expert_idx] = {
                        "total_selections": total_selections,
                        "selection_percentage": (
                            total_selections / (seq_len * block.moe.top_k)
                        )
                        * 100,
                        "avg_gate_value": avg_gate_value,
                        "capacity_fulfillment": capacity_fulfillment,
                    }

                    # Update load balancing metrics
                    if total_selections == 0:
                        layer_stats[i]["load_balancing"]["unused_expert_count"] += 1
                    elif capacity_fulfillment > 1.5:  # Over 150% of ideal load
                        layer_stats[i]["load_balancing"]["overflow_expert_count"] += 1

                # Calculate load balancing metrics
                expert_loads = torch.tensor(
                    [
                        stats["total_selections"]
                        for stats in layer_stats[i]["expert_stats"].values()
                    ],
                    dtype=torch.float32,
                )

                # Calculate coefficient of variation (CV) as a measure of load balance
                cv = (
                    torch.std(expert_loads) / torch.mean(expert_loads)
                    if torch.mean(expert_loads) > 0
                    else torch.tensor(0.0)
                )
                layer_stats[i]["load_balancing"]["coefficient_of_variation"] = cv.item()

            # Process through the block
            if hasattr(block, "moe"):
                x, _ = block(x)
            else:
                x = block(x)

    return layer_stats


def print_expert_analysis(stats: dict):
    """Pretty print the expert analysis statistics."""
    print("\nDetailed Expert Utilization Analysis:")
    print("=" * 80)

    for layer_idx, layer_data in stats.items():
        print(f"\nLayer {layer_idx}:")
        print("-" * 40)

        # Print expert-specific stats
        print("\nExpert Statistics:")
        for expert_idx, expert_stats in layer_data["expert_stats"].items():
            print(f"\nExpert {expert_idx}:")
            print(f"  Selections: {expert_stats['total_selections']} tokens")
            print(f"  Selection %: {expert_stats['selection_percentage']:.2f}%")
            print(f"  Avg Gate Value: {expert_stats['avg_gate_value']:.4f}")
            print(
                f"  Capacity Fulfillment: {expert_stats['capacity_fulfillment']:.2f}x ideal load"
            )

        # Print load balancing metrics
        print("\nLoad Balancing Metrics:")
        lb_stats = layer_data["load_balancing"]
        print(f"  Coefficient of Variation: {lb_stats['coefficient_of_variation']:.4f}")
        print(f"  Unused Experts: {lb_stats['unused_expert_count']}")
        print(f"  Overloaded Experts: {lb_stats['overflow_expert_count']}")

        print("\nSequence Info:")
        print(f"  Sequence Length: {layer_data['seq_length']}")
        print(f"  Number of Experts: {layer_data['num_experts']}")
        print(f"  Top-k: {layer_data['top_k']}")


def main():
    init()  # Initialize colorama
    parser = argparse.ArgumentParser(
        description=f"{Fore.CYAN}Test MoE GPT model{Style.RESET_ALL}"
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a time",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--instruction_mode", action="store_true", help="Use Alpaca instruction format"
    )
    parser.add_argument(
        "--system_prompt", type=str, help="Optional system prompt for instruction mode"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--analyze_experts", action="store_true", help="Analyze expert utilization"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="gpt2",
        choices=["gpt2", "cl100k_base"],
        help="Tiktoken encoding to use",
    )
    parser.add_argument(
        "--is_moe",
        action="store_true",
        help="Enable pre-training mode where both pathways predict next tokens",
    )
    args = parser.parse_args()

    # Load model with inferred config
    device = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{Fore.GREEN}Loading model on {device}...{Style.RESET_ALL}")
    model = load_model(args.checkpoint)
    model.to(device)

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(args.encoding)
    if tokenizer.n_vocab != model.config.vocab_size:
        print(
            f"\n{Fore.YELLOW}Warning: Tokenizer vocab size ({tokenizer.n_vocab}) doesn't match model vocab size ({model.config.vocab_size})"
        )
        print(
            f"You might want to use a different tiktoken encoding that matches your model's vocabulary.{Style.RESET_ALL}"
        )

    # Initialize Alpaca formatter if using instruction mode
    formatter = None
    if args.instruction_mode:
        formatter = AlpacaFormatter()
        print(f"\n{Fore.BLUE}Using Alpaca instruction format{Style.RESET_ALL}")

    if args.analyze_experts:
        print(f"\n{Fore.BLUE}Analyzing expert utilization...{Style.RESET_ALL}")
        stats = analyze_expert_usage_detailed(model, args.prompt, tokenizer, device)
        print_expert_analysis(stats)

    # Generate text
    print(f"\n{Fore.CYAN}Generating text...{Style.RESET_ALL}")
    generated_text = generate_sequence(
        model,
        tokenizer,
        args.prompt,
        formatter=formatter,
        instruction_mode=args.instruction_mode,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print(f"\n{Fore.MAGENTA}Prompt:{Style.RESET_ALL} {args.prompt}")
    print(f"{Fore.GREEN}Generated text:{Style.RESET_ALL} {generated_text}")


if __name__ == "__main__":
    main()
