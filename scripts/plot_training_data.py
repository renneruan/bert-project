import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

WORK_DIR = os.getenv("WORK")
PARENT_LOG_DIR = f"{WORK_DIR}/training_test/Modern/4.6/runs/"

# Resgata o último experimento no diretório de runs
try:
    all_runs = [
        os.path.join(PARENT_LOG_DIR, d)
        for d in os.listdir(PARENT_LOG_DIR)
        if os.path.isdir(os.path.join(PARENT_LOG_DIR, d))
    ]
    latest_log_dir = max(all_runs, key=os.path.getmtime)
    print(f"Latest experiment directory: {latest_log_dir}")
except (ValueError, FileNotFoundError):
    print(
        f"Error: No subdirectories found in '{PARENT_LOG_DIR}'. Please check the path."
    )
    exit()

# Resgata último evento do experimento geralmente se começa com 'events.out.tfevents.'
try:
    event_file = os.path.join(
        latest_log_dir,
        [
            f
            for f in os.listdir(latest_log_dir)
            if f.startswith("events.out.tfevents")
        ][0],
    )
except IndexError:
    print(f"Error: No event file found in '{latest_log_dir}'")
    exit()


ea = event_accumulator.EventAccumulator(
    event_file, size_guidance={event_accumulator.SCALARS: 0}
)
ea.Reload()

# Lista escalares disponíveis (e.g., 'Loss/train', 'Accuracy/val')
print("Available scalar tags:", ea.Tags()["scalars"])


def get_scalar_dataframe(tag_name):
    """Extrai dados escalares de uma tag para um pandas DataFrame."""
    try:
        events = ea.Scalars(tag_name)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        return pd.DataFrame({"step": steps, "value": values})
    except KeyError:
        print(f"Warning: Tag '{tag_name}' not found.")
        return None


train_loss_df = get_scalar_dataframe("train/loss")
grad_norm = get_scalar_dataframe("train/grad_norm")

# Seção de plots dos dados
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 7))

if train_loss_df is not None:
    ax.plot(
        train_loss_df["step"],
        train_loss_df["value"],
        label="Training Loss",
        color="dodgerblue",
        alpha=0.9,
    )
if grad_norm is not None:
    ax.plot(
        grad_norm["step"],
        grad_norm["value"],
        label="Grad Norm",
        color="orangered",
        linestyle="--",
    )

ax.set_title("Training and Grad Norm Over Time", fontsize=16)
ax.set_xlabel("Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.legend(fontsize=12)
ax.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.tight_layout()

output_filename = "training_loss_plot.pdf"
plt.savefig(output_filename, format="pdf", bbox_inches="tight")

print(f"Plot saved successfully as '{output_filename}'")
