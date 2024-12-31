import torch
import torch.nn as nn

class LatentLoopTransformer(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer = transformer_model
        self.hidden_size = transformer_model.config.hidden_size
        self.bot_token_id = 1  # Define special token IDs
        self.eot_token_id = 2

    def forward(self, input_ids, mode='language', max_loops=5):
        # Embedding lookup
        embeddings = self.transformer.embeddings(input_ids)
        hidden_states = embeddings
        loop_count = 0
        outputs = []

        while loop_count < max_loops:
            if mode == 'latent':
                # Pass through transformer without decoding
                hidden_states = self.transformer.encoder(hidden_states)
                last_hidden_state = hidden_states[:, -1, :]  # Last hidden state
                
                # Append last hidden state to sequence
                hidden_states = torch.cat([hidden_states, last_hidden_state.unsqueeze(1)], dim=1)
                loop_count += 1

                # Check for <eot> token in input_ids to switch to language mode
                if (input_ids[:, -1] == self.eot_token_id).any():
                    mode = 'language'

            elif mode == 'language':
                # Generate tokens in language mode
                logits = self.transformer.lm_head(hidden_states)
                next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                outputs.append(next_token)

                # Update input_ids and embeddings
                input_ids = torch.cat([input_ids, next_token], dim=1)
                hidden_states = self.transformer.embeddings(input_ids)

                # Stop if EOS token is generated
                if next_token.item() == self.transformer.config.eos_token_id:
                    break

        return torch.cat(outputs, dim=1)