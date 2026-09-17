# experiments/01_train_baseline.py
# Command to run:
# python -m experiments.01_train_baseline

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import load_cora
from src.model import GCN
from src.utils import set_seed


# ============================================================
# Experiment configuration
# ============================================================

SEED = 42
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
HIDDEN_CHANNELS = 16
DROPOUT = 0.5


def accuracy(logits, labels):
    """
    Compute classification accuracy from model logits.
    """
    predictions = logits.argmax(dim=-1)

    return (
        predictions.eq(labels)
        .float()
        .mean()
        .item()
    )


def main():
    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset, data = load_cora()

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    data = data.to(device)

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = GCN(
        in_channels=dataset.num_features,
        hidden_channels=HIDDEN_CHANNELS,
        out_channels=dataset.num_classes,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Best-validation checkpoint tracking
    # --------------------------------------------------------

    best_val_acc = -1.0
    best_epoch = -1
    best_model_state = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n=== Training Baseline GCN ===")

    for epoch in range(1, EPOCHS + 1):

        # ----------------------------------------------------
        # Training step
        # ----------------------------------------------------

        model.train()

        optimizer.zero_grad()

        logits = model(
            data.x,
            data.edge_index,
        )

        loss = F.cross_entropy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        loss.backward()
        optimizer.step()

        # ----------------------------------------------------
        # Validation evaluation
        #
        # Evaluate every epoch so that model selection does
        # not depend only on the epochs printed to terminal.
        # ----------------------------------------------------

        model.eval()

        with torch.no_grad():

            eval_logits = model(
                data.x,
                data.edge_index,
            )

            train_acc = accuracy(
                eval_logits[data.train_mask],
                data.y[data.train_mask],
            )

            val_acc = accuracy(
                eval_logits[data.val_mask],
                data.y[data.val_mask],
            )

        # ----------------------------------------------------
        # Retain best validation checkpoint
        #
        # Strict ">" means that, in the event of a tie,
        # the earliest best epoch is retained.
        # ----------------------------------------------------

        if val_acc > best_val_acc:

            best_val_acc = val_acc
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

        # ----------------------------------------------------
        # Periodic terminal logging
        # ----------------------------------------------------

        if epoch == 1 or epoch % 20 == 0:

            print(
                f"Epoch {epoch:03d} | "
                f"Loss {loss.item():.4f} | "
                f"Train Acc {train_acc:.4f} | "
                f"Val Acc {val_acc:.4f}"
            )

    # --------------------------------------------------------
    # Restore best-validation model
    # --------------------------------------------------------

    if best_model_state is None:
        raise RuntimeError(
            "No best model state was recorded during training."
        )

    model.load_state_dict(best_model_state)
    model.eval()

    # --------------------------------------------------------
    # Final evaluation using restored best model
    #
    # The test set is evaluated only after model selection.
    # --------------------------------------------------------

    with torch.no_grad():

        logits = model(
            data.x,
            data.edge_index,
        )

        train_acc = accuracy(
            logits[data.train_mask],
            data.y[data.train_mask],
        )

        val_acc = accuracy(
            logits[data.val_mask],
            data.y[data.val_mask],
        )

        test_acc = accuracy(
            logits[data.test_mask],
            data.y[data.test_mask],
        )

    # --------------------------------------------------------
    # Report results
    # --------------------------------------------------------

    print("\n=== Best Validation Checkpoint ===")
    print(f"Best epoch:          {best_epoch}")
    print(f"Best validation acc: {best_val_acc:.4f}")

    print("\n=== Final Baseline Results ===")
    print(f"Train accuracy:      {train_acc:.4f}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")

    # --------------------------------------------------------
    # Save frozen checkpoint
    # --------------------------------------------------------

    Path("models").mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = Path(
        "models/cora_gcn_seed42.pt"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": SEED,
            "epochs": EPOCHS,
            "best_epoch": best_epoch,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "hidden_channels": HIDDEN_CHANNELS,
            "dropout": DROPOUT,
            "num_features": dataset.num_features,
            "num_classes": dataset.num_classes,
            "train_accuracy": train_acc,
            "validation_accuracy": val_acc,
            "best_validation_accuracy": best_val_acc,
            "test_accuracy": test_acc,
        },
        checkpoint_path,
    )

    print(
        f"\nCheckpoint saved to: {checkpoint_path}"
    )


if __name__ == "__main__":
    main()
