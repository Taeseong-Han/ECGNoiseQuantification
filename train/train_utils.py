import copy
import torch

from tqdm import tqdm
from pathlib import Path
from accelerate import Accelerator


def _ema_params_to_state_dict(model: torch.nn.Module, ema_params: list) -> dict:
    """
    Convert a list of EMA parameters to a state_dict compatible with torch.save.
    """
    state_dict = model.state_dict()
    for (name, _), param in zip(model.named_parameters(), ema_params):
        state_dict[name] = param
    return state_dict


def update_ema(target_params: list, source_params: list, rate: float = 0.99) -> None:
    """
    Update target parameters towards source parameters using exponential moving average.
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def save_checkpoint(
        model: torch.nn.Module,
        ema_params: list,
        output_dir: Path,
        epoch: int
) -> None:
    """
    Save model and EMA model weights to the specified directory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = output_dir / f"{epoch}.pth"
    ema_file = output_dir / f"ema_{epoch}.pth"

    torch.save(model.state_dict(), model_file)
    torch.save(_ema_params_to_state_dict(model, ema_params), ema_file)


def train_loop(
        args,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        lr_scheduler
) -> None:
    """
    Unified training loop for autoencoder or diffusion models using HuggingFace Accelerate.

    Args:
        args: Configuration object with attributes:
            - seed
            - mixed_precision
            - gradient_accumulation_steps
            - log_dir
            - save_path
            - num_epochs
            - ema_rate
        model: Model to train.
        optimizer: Optimizer for training.
        dataloader: DataLoader providing training batches.
        lr_scheduler: Learning rate scheduler (optional).
    """
    # Set seeds and initialize accelerator
    torch.manual_seed(args.seed)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=str(args.log_dir),
    )

    # Create directories
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Initialize trackers
    accelerator.init_trackers(Path(args.save_path).name)

    # Prepare for distributed/progress
    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    # Initialize EMA parameters
    ema_params = copy.deepcopy(list(model.parameters()))
    save_interval = max(1, args.num_epochs // 10)
    global_step = 0

    for epoch in range(1, args.num_epochs + 1):
        progress_bar = tqdm(
            total=len(dataloader),
            disable=not accelerator.is_local_main_process,
            desc=f"Epoch {epoch}"
        )

        for batch in dataloader:
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)

                # Update optimizer and scheduler
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if lr_scheduler:
                    lr_scheduler.step()
                optimizer.zero_grad()

                # EMA update
                update_ema(ema_params, list(model.parameters()), args.ema_rate)

            # Logging and progress
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "step": global_step}
            if lr_scheduler:
                logs["lr"] = lr_scheduler.get_last_lr()[0]

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        progress_bar.close()

        # Save checkpoints periodically
        if epoch % save_interval == 0 or epoch == args.num_epochs:
            save_checkpoint(model, ema_params, Path(args.save_path), epoch)
