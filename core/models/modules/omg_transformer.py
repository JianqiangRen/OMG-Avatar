
import torch
import torch.nn as nn

class TransformerDecoderWithProjection(nn.Module):
    def __init__(
        self,
        d_model=256,               # Decoder feature dimension (matches tgt input)
        nhead=8,                   # Number of attention heads
        num_layers=6,              # Number of decoder layers
        memory_dim=768,            # Original memory input dimension (from encoder)
        input_dim = 1024,
        dim_feedforward=2048,      # FFN hidden dimension
        dropout=0.1                # Dropout rate
    ):
        super().__init__()

        # Projection layers: project inputs to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.memory_projection = nn.Linear(memory_dim, d_model)

        # Build a single Transformer decoder layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # Input shape: (batch_size, seq_len, d_model)
        )

        # Stack multiple decoder layers
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers
        )

    def forward(
        self,
        vert_feature: torch.Tensor,                         # shape: (batch_size, 5023, 256)
        image_feature: torch.Tensor                       # shape: (batch_size, 1369, 768)
    ) -> torch.Tensor:
        """
        :param tgt: Decoder input (target sequence)
        :param memory: Encoder output (context information)
        :return: Decoder output (shape: (batch_size, 5023, 256))
        """
        # Project memory to d_model dimension
        projected_vert_feature = self.input_projection(vert_feature)
        projected_memory = self.memory_projection(image_feature)  # shape: (batch_size, 1369, 256)

        # Run the Transformer decoder
        output = self.decoder(
            tgt=projected_vert_feature,
            memory=projected_memory
        )
        return output
