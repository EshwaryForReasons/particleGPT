import wandb
import numpy as np
import matplotlib.pyplot as plt

# 1. Initialize the project
wandb.init(
    project="particleGPT",
    name="run-001",  # optional: unique run name
    config={
        "learning_rate": 1e-4,
        "schedule": "cosine",
        "tokenizer": "custom-particle-tokenizer-v2",
        "vocab_size": 3000,
        "trainable_params": 42_000_000
    }
)

# 2. Log a training metric
for step in range(10):
    wandb.log({"train_loss": np.exp(-step/5), "step": step})

# 3. Log a physics distribution
dist = np.random.normal(0, 1, 1000)  # example distribution
fig, ax = plt.subplots()
ax.hist(dist, bins=50)
wandb.log({"physics_distribution": wandb.Image(fig)})

# 4. Save raw distribution as artifact
np.save("distribution.npy", dist)
wandb.save("distribution.npy")  # Upload file to W&B
