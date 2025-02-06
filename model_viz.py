import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Arc, ConnectionPatch
import matplotlib.colors as mcolors

class AttentionVisualizer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def get_attention_maps(self, input_text, layer_idx=None):
        """Extract attention weights from the model for given input"""
        self.model.eval()
        tokens = torch.tensor([self.tokenizer.encode(input_text)]).to(next(self.model.parameters()).device)
        
        # Store attention weights
        attention_maps = []
        
        def attention_hook(module, input, output):
            # Get attention weights from output
            # Shape: (batch_size, num_heads, seq_len, seq_len)
            attention_maps.append(output[1].detach())
            
        # Register hooks
        hooks = []
        if layer_idx is not None:
            hooks.append(self.model.blocks[layer_idx].attn.register_forward_hook(attention_hook))
        else:
            for block in self.model.blocks:
                hooks.append(block.attn.register_forward_hook(attention_hook))
        
        # Forward pass
        with torch.no_grad():
            self.model(tokens)
            
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return attention_maps, self.tokenizer.decode(tokens[0].tolist())

    def plot_attention_heatmap(self, input_text, layer_idx=0, head_idx=0, save_path=None):
        """Plot attention heatmap for a specific layer and head"""
        attention_maps, text = self.get_attention_maps(input_text, layer_idx)
        attention_weights = attention_maps[0][0, head_idx].cpu().numpy()
        
        # Get tokens for labels
        tokens = self.tokenizer.encode(input_text)
        token_labels = [self.tokenizer.decode([t]) for t in tokens]
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(attention_weights, 
                   xticklabels=token_labels,
                   yticklabels=token_labels,
                   cmap='viridis',
                   square=True)
        
        plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_attention_flow(self, input_text, layer_idx=0, head_idx=0, threshold=0.1, save_path=None):
        """Plot attention flow diagram showing token relationships"""
        attention_maps, text = self.get_attention_maps(input_text, layer_idx)
        attention_weights = attention_maps[0][0, head_idx].cpu().numpy()
        
        tokens = self.tokenizer.encode(input_text)
        token_labels = [self.tokenizer.decode([t]) for t in tokens]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlim(0, len(tokens))
        ax.set_ylim(-1, 1)
        
        # Plot tokens
        token_positions = np.arange(len(tokens))
        ax.scatter(token_positions, np.zeros_like(token_positions), c='blue', s=100)
        
        # Add token labels
        for i, label in enumerate(token_labels):
            ax.annotate(label, (i, -0.1), ha='center', va='top')
        
        # Draw attention flows
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                weight = attention_weights[i, j]
                if weight > threshold:
                    # Calculate arc height based on distance and weight
                    distance = abs(i - j)
                    height = 0.5 * weight * min(distance / len(tokens), 0.5)
                    
                    # Create arc
                    arc = Arc((i + j) / 2, 0, 
                            width=distance,
                            height=height * 2,
                            angle=0,
                            theta1=0,
                            theta2=180,
                            alpha=weight,
                            color='red')
                    ax.add_patch(arc)
        
        ax.set_title(f'Attention Flow (Layer {layer_idx}, Head {head_idx})')
        ax.set_xticks([])
        ax.set_yticks([])
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def plot_multi_head_attention(self, input_text, layer_idx=0, save_path=None):
        """Plot attention patterns for all heads in a layer"""
        attention_maps, text = self.get_attention_maps(input_text, layer_idx)
        num_heads = attention_maps[0].shape[1]
        
        # Calculate grid dimensions
        grid_size = int(np.ceil(np.sqrt(num_heads)))
        
        # Create figure
        fig, axes = plt.subplots(grid_size, grid_size, 
                                figsize=(4*grid_size, 4*grid_size))
        axes = axes.flatten()
        
        tokens = self.tokenizer.encode(input_text)
        token_labels = [self.tokenizer.decode([t]) for t in tokens]
        
        for head_idx in range(num_heads):
            if head_idx < len(axes):
                attention_weights = attention_maps[0][0, head_idx].cpu().numpy()
                
                sns.heatmap(attention_weights,
                           xticklabels=token_labels if head_idx >= (len(axes) - grid_size) else [],
                           yticklabels=token_labels if head_idx % grid_size == 0 else [],
                           cmap='viridis',
                           square=True,
                           ax=axes[head_idx],
                           cbar=False)
                
                axes[head_idx].set_title(f'Head {head_idx}')
                
                # Rotate labels for better readability
                if head_idx >= (len(axes) - grid_size):
                    axes[head_idx].set_xticklabels(axes[head_idx].get_xticklabels(),
                                                 rotation=45, ha='right')
        
        # Remove empty subplots
        for idx in range(num_heads, len(axes)):
            fig.delaxes(axes[idx])
            
        plt.suptitle(f'Multi-Head Attention Patterns (Layer {layer_idx})',
                    fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def animate_attention_across_layers(self, input_text, head_idx=0, save_path=None):
        """Create an animation of attention patterns across layers"""
        import matplotlib.animation as animation
        
        attention_maps, text = self.get_attention_maps(input_text)
        tokens = self.tokenizer.encode(input_text)
        token_labels = [self.tokenizer.decode([t]) for t in tokens]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        plt.close()  # Prevent display of empty figure
        
        def update(frame):
            ax.clear()
            attention_weights = attention_maps[frame][0, head_idx].cpu().numpy()
            
            sns.heatmap(attention_weights,
                       xticklabels=token_labels,
                       yticklabels=token_labels,
                       cmap='viridis',
                       square=True,
                       ax=ax)
            
            ax.set_title(f'Attention Pattern (Layer {frame}, Head {head_idx})')
            plt.xticks(rotation=45, ha='right')
            
        anim = animation.FuncAnimation(fig, update,
                                     frames=len(attention_maps),
                                     interval=1000)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        return anim
    

"""
model = YourGPTModel()
tokenizer = YourTokenizer()
visualizer = AttentionVisualizer(model, tokenizer)

# Plot single head attention heatmap
visualizer.plot_attention_heatmap(
    "The quick brown fox jumps over the lazy dog",
    layer_idx=0,
    head_idx=0,
    save_path="attention_heatmap.png"
)

# Plot attention flow
visualizer.plot_attention_flow(
    "The quick brown fox jumps over the lazy dog",
    layer_idx=0,
    head_idx=0,
    save_path="attention_flow.png"
)

# Plot all heads in a layer
visualizer.plot_multi_head_attention(
    "The quick brown fox jumps over the lazy dog",
    layer_idx=0,
    save_path="multi_head_attention.png"
)

# Create animation across layers
visualizer.animate_attention_across_layers(
    "The quick brown fox jumps over the lazy dog",
    head_idx=0,
    save_path="attention_animation.gif"
)
"""