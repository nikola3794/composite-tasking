import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

from ..system import System


class SingleTaskingSystem(System):
    def __init__(self, cfg, task_to_id_dict, colourmaps, task_z_code_dict):
        super().__init__(
            cfg=cfg, 
            task_to_id_dict=task_to_id_dict,
            colourmaps=colourmaps,
            task_z_code_dict=task_z_code_dict
        )

        # This system will have a manually implemented backrpop
        self.automatic_optimization = False

    def _one_step(self, batch, batch_idx, which_split):
        """
        Does one batch forward pass and computes the losses.
        Metrics are updated after the step ends.
        """

        # TODO Compute outside this function and pass as an argument
        # TODO for both loss and metric
        # Extract all tasks used in the batch
        used_tasks = torch.sum(batch["used_tasks"], axis=0)
        considered_task_ids = torch.nonzero(torch.abs(used_tasks), as_tuple=True)
        considered_task_ids = considered_task_ids[0].tolist()

        # Calculate the loss, which consists of terms for each considered task
        pred_logits = {}
        preds = {}
        loss_dict = {}
        loss_total = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        for task_id in considered_task_ids:
            # Forward pass
            pred_logits, preds = self.forward(batch=batch, task_id=task_id)

            # Compute losses
            loss_id = self._compute_loss(
                pred_logits=pred_logits,
                preds=preds,
                batch=batch,
                task_id=task_id
            )

            # Accumulate total loss
            # In this case the total loss is the selected task loss
            loss_total += loss_id

            # Total loss
            loss_dict["loss_total"] = loss_id

            # Manual backrpoagation for train batches
            # If a loss was NaN, a 0 will be put maunally 
            # (which does not require gradients)
            if which_split == "train" and loss_id.requires_grad:
                self._manual_backprop(loss=loss_id)
        
        return {
            "preds": preds,
            "pred_logits": pred_logits,
            "batch": batch,
            "loss_dict": loss_dict
        } 
    
    def forward(self, batch, task_id):

        pred_logits = self.model.forward_single_task(
            x=batch["image"], 
            task_id=task_id
        )
        pred = torch.sigmoid(pred_logits)

        return pred_logits, pred
        
    # If there is need for a very specific loss, 
    # this method can be re-defined in the child class
    def _compute_loss(self, pred_logits, preds, batch, task_id):
        """
        Calculate losses for the current task
        """

        # Considered task
        task = self.id_to_task_dict[task_id]

        shape = pred_logits.shape
        select_mask = task_id * torch.ones(shape, dtype=torch.uint8, device=self.device)
        select_mask = (select_mask == task_id)

        # Call the loss function for the current task
        loss_args = {
            "pred": preds if task in ["seg", "parts"] else pred_logits, 
            "target": batch["label"], 
            "select_mask": select_mask,
        }
        if task == "parts":
            loss_args["valid_imgs"] = batch["valid_parts_img"]
        loss_id = self.losses[task](**loss_args)

        # Defense against possible NaN's during training
        if torch.isnan(loss_id):
            print(f"The loss component for task -{task}- was nan and was discarded")
            loss_id = torch.tensor(0.0, device=self.device)

        # Scale the loss with the weight for the current task
        loss_id = loss_id * self.training_cfg[f"{task}_l_w"]

        return loss_id

    def _update_metrics_step_end(self, pred_logits, preds, batch):
        """
        Metric update which happens at every step
        """

        # TODO Compute outside this function and pass as an argument
        # TODO for both loss and metric
        # Extract all tasks used in the batch
        used_tasks = torch.sum(batch["used_tasks"], axis=0)
        considered_task_ids = torch.nonzero(torch.abs(used_tasks), as_tuple=True)
        considered_task_ids = considered_task_ids[0].tolist()

        # Calculate the metrics for each considered task
        for task_id in considered_task_ids:
            task = self.id_to_task_dict[task_id]

            shape = pred_logits.shape
            select_mask = task_id * torch.ones(shape, dtype=torch.uint8, device=self.device)
            select_mask = (select_mask == task_id)

            with torch.no_grad():            
                # Call the metric for the current task
                metric_args = {
                    "pred": pred_logits if task == "normals" else preds, 
                    "target": batch["label"], 
                    "select_mask": select_mask,
                }
                if task == "parts":
                    metric_args["valid_parts_img"] = batch["valid_parts_img"]
                self.metrics[task].update(**metric_args)

    def _get_optim_dict(self):
        # Different learning rate for pre-trained encoder (slower learning)
        # and fore the decoder with conditioning modules (faster learning)
        optim_list = []
        optim_list.append(
            {
                "params": self.model.get_encoder_parameters(), 
                "lr": self.training_cfg["lr"] / self.training_cfg["lr_div_encoder"], 
                "weight_decay": self.training_cfg["wd"]
            }
        )
        optim_list.append(
            {
                "params": self.model.get_decoder_parameters(),
                "lr": self.training_cfg["lr"], 
                "weight_decay": self.training_cfg["wd"]
            }
        )

        return optim_list
