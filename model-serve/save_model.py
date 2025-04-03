import bentoml
from model import EmojiClassifier

model = EmojiClassifier.load_from_checkpoint("checkpoints/best_model.ckpt")

tag = bentoml.pytorch_lightning.save_model(
    "emoji_classifier", model, metadata={"num_classes": 18, "lr": 1e-3}
)

print(f"Model saved with tag: {tag}")
