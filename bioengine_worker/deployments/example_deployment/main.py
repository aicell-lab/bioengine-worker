class ExampleModel(object):
    def __init__(self):
        pass

    async def _get_model(self, model_id: str):
        """Load the model. The parameter `model_id` is required for a `ray.serve.multiplexed` method"""
        import torch.nn as nn

        # Initialize transformer directly
        model = nn.Transformer(
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            batch_first=True,
        )

        return model

    async def ping(self) -> str:
        return "pong"

    async def train(self, model_id="dummy", data=None) -> dict:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        model = await self._get_model(model_id)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(model.parameters())
        criterion = nn.MSELoss()

        # Create dummy data for demonstration
        batch_size, seq_len = 2, 10
        d_model = 64

        # Generate random input and target sequences
        src = torch.randn(batch_size, seq_len, d_model).to(device)
        tgt = torch.randn(batch_size, seq_len, d_model).to(device)

        # Training step
        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()

        return {
            "loss": loss.item(),
            "message": "Completed one training iteration",
            "data": data,
        }
