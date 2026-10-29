import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        logits = self.classifier(x)
        return logits


def validate_model_shapes() -> None:
    model = CNNClassifier()
    dummy_batch = torch.randn(4, 3, 32, 32)
    output = model(dummy_batch)

    print("Input shape:", dummy_batch.shape)
    print("After Conv blocks shape:", model.features(dummy_batch).shape)
    print("Output shape:", output.shape)
    assert dummy_batch.ndim == 4, "La entrada debe ser un tensor 4D [B, C, H, W]."
    assert output.shape == (4, 10), "El modelo debe producir 10 logits por imagen."
    print("Validación de shapes: OK")


if __name__ == "__main__":
    validate_model_shapes()
