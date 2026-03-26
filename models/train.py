import torch
import numpy as np
import os
import sys
import toml
import optuna

wandb = None  # not used

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from dataset.utils import setup_dataloaders
from models.inp import INP
from models.loss import ELBOLoss

EVAL_ITER = 500
SAVE_ITER = 500
MAX_EVAL_IT = 50


class Trainer:
    def __init__(self, config, save_dir, load_path=None, last_save_it=0):
        self.config = config
        self.last_save_it = last_save_it

        self.device = config.device
        self.train_dataloader, self.val_dataloader, _, extras = setup_dataloaders(
            config
        )

        for k, v in extras.items():
            config.__dict__[k] = v

        self.num_epochs = config.num_epochs

        self.model = self._create_model(config)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)

        self.loss_func = ELBOLoss(beta=config.beta)
        if load_path is not None:
            print(f"Loading model from state dict {load_path}")
            state_dict = torch.load(load_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            loaded_states = set(state_dict.keys())

        own_trainable_states = []
        print("Trainable parameters:")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name)
                own_trainable_states.append(name)

        if load_path is not None:
            own_trainable_states = set(own_trainable_states)
            print("\n States not loaded from state dict:")
            print(
                *sorted(list(own_trainable_states.difference(loaded_states))), sep="\n"
            )
            print("Unknown states:")
            print(
                *sorted(
                    list(loaded_states.difference(set(self.model.state_dict().keys())))
                ),
                sep="\n",
            )

        self.save_dir = save_dir

    def _create_model(self, config):
        """Create the model. Override in subclasses for different model types."""
        return INP(config)

    def get_loss(self, x_context, y_context, x_target, y_target, knowledge, knowledge_target=None):
        if self.config.sort_context:
            x_context, indices = torch.sort(x_context, dim=1)
            y_context = torch.gather(y_context, 1, indices)
        if self.config.use_knowledge:
            output = self.model(
                x_context,
                y_context,
                x_target,
                y_target=y_target,
                knowledge=knowledge,
                knowledge_target=knowledge_target,
            )
        else:
            output = self.model(
                x_context, y_context, x_target, y_target=y_target, knowledge=None
            )
        loss, kl, negative_ll = self.loss_func(output, y_target)

        results = {"loss": loss, "kl": kl, "negative_ll": negative_ll}

        return results

    def run_batch_train(self, batch):
        context, target, knowledge, ids = batch
        x_context, y_context = context
        x_target, y_target = target
        x_context = x_context.to(self.device)
        y_context = y_context.to(self.device)
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        results = self.get_loss(x_context, y_context, x_target, y_target, knowledge)

        return results

    def _after_backward(self):
        """Hook called after loss.backward(), before optimizer.step().
        Override in subclasses for gradient clipping etc."""
        pass

    def run_batch_eval(self, batch, num_context=5):
        context, target, knowledge, ids = batch
        x_target, y_target = target
        x_target = x_target.to(self.device)
        y_target = y_target.to(self.device)

        context_idx = np.random.choice(x_target.shape[1], num_context, replace=False)

        x_context, y_context = x_target[:, context_idx, :], y_target[:, context_idx, :]

        results = self.get_loss(x_context, y_context, x_target, y_target, knowledge)

        return results

    def train(self):
        it = 0
        min_eval_loss = np.inf
        patience = getattr(self.config, "early_stopping_patience", 0)
        evals_without_improvement = 0
        for epoch in range(self.num_epochs + 1):
            # self.scheduler.step()
            early_stop = False
            for batch in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                results = self.run_batch_train(batch)
                loss = results["loss"]
                kl = results["kl"]
                negative_ll = results["negative_ll"]
                loss.backward()
                self._after_backward()
                self.optimizer.step()
                if wandb is not None and wandb.run is not None:
                    wandb.log({"train_loss": loss, "train_negative_ll": negative_ll, "train_kl": kl})

                if it % EVAL_ITER == 0 and it > 0:
                    losses, val_loss = self.eval()
                    mean_eval_loss = np.mean(list(losses.values()))
                    if wandb is not None and wandb.run is not None:
                        wandb.log({"mean_eval_loss": mean_eval_loss, "eval_loss": val_loss})
                        for k, v in losses.items():
                            wandb.log({f"eval_loss_{k}": v})

                    if val_loss < min_eval_loss:
                        min_eval_loss = val_loss
                        evals_without_improvement = 0
                        torch.save(
                            self.model.state_dict(), f"{self.save_dir}/model_best.pt"
                        )
                        torch.save(
                            self.optimizer.state_dict(),
                            f"{self.save_dir}/optim_best.pt",
                        )
                        print(f"Best model saved at iteration {self.last_save_it + it}")
                    else:
                        evals_without_improvement += 1

                    if patience > 0 and evals_without_improvement >= patience:
                        print(f"Early stopping at iteration {it} (no improvement for {patience} evals)")
                        early_stop = True
                        break

                it += 1
            if early_stop:
                break

        return min_eval_loss

    def eval(self):
        print("Evaluating")
        it = 0
        self.model.eval()
        with torch.no_grad():
            loss_num_context = [3, 5, 10]
            if self.config.min_num_context == 0:
                loss_num_context = [0] + loss_num_context
            losses_dict = dict(zip(loss_num_context, [[] for _ in loss_num_context]))

            val_losses = []
            for batch in self.val_dataloader:
                for num_context in loss_num_context:
                    results = self.run_batch_eval(batch, num_context=num_context)
                    loss = results["loss"]
                    val_results = self.run_batch_train(batch)
                    val_loss = val_results["loss"]
                    losses_dict[num_context].append(loss.to("cpu").item())
                    val_losses.append(val_loss.to("cpu").item())

                it += 1
                if it > MAX_EVAL_IT:
                    break
            losses_dict = {k: np.mean(v) for k, v in losses_dict.items()}
            val_loss = np.mean(val_losses)

        return losses_dict, val_loss


def get_device():
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        device = torch.device("cuda:{}".format(0))
    else:
        device = "cpu"
    print("Using device: {}".format(device))
    return device


def meta_train(trial, config, run_name_prefix="run"):
    device = get_device()
    config.device = device

    # Create save folder and save config
    save_dir = f"./saves/{config.project_name}"
    os.makedirs(save_dir, exist_ok=True)

    save_no = len(os.listdir(save_dir))
    save_no = [
        int(x.split("_")[-1])
        for x in os.listdir(save_dir)
        if x.startswith(run_name_prefix)
    ]
    if len(save_no) > 0:
        save_no = max(save_no) + 1
    else:
        save_no = 0
    save_dir = f"{save_dir}/{run_name_prefix}_{save_no}"
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(config=config, save_dir=save_dir)

    config = trainer.config

    # save config
    config.write_config(f"{save_dir}/config.toml")

    if wandb is not None:
        wandb.init(
            project=config.project_name,
            name=f"{run_name_prefix}_{save_no}",
            config=vars(config),
        )
    best_eval_loss = trainer.train()
    if wandb is not None and wandb.run is not None:
        wandb.finish()

    return best_eval_loss


if __name__ == "__main__":
    # resume_training('run_7')
    import random
    import numpy as np
    from config import Config

    # read config from config.toml
    config = toml.load("config.toml")
    config = Config(**config)

    # set seed
    seed = config.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # begin study
    study = optuna.create_study(direction="minimize")

    study.optimize(
        lambda x: meta_train(x, config=config, run_name_prefix=config.run_name_prefix),
        n_trials=config.n_trials,
    )
