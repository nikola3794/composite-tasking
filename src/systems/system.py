import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl

import torchmetrics

from ..losses.balanced_bce import BalancedBCELoss
from ..losses.cosine_similarity_loss import CosineSImilarityLoss
from ..losses.focal_ce_grid_to_seg import FocalCEGridToSeg
from ..losses.l1_loss_discard_zero import L1LossDiscardZero

from ..metrics.normals_metric import NormalsMetric
from ..metrics.edges_sal_metric import EdgesSaliencyMetric
from ..metrics.seg_parts_metric import SegMetric

from ..models.composite_tasking_net_v1 import CompositeTaskingNetV1
from ..models.multi_head_net_v1 import MultiHeadNetV1
from ..models.multi_net_v1 import MultiNetV1
from ..models.single_tasking_net_v1 import SingleTaskingNetV1


class System(pl.LightningModule):

    def __init__(self, cfg, task_to_id_dict, colourmaps, task_z_code_dict):
        super().__init__()

        # Store input arguments
        self.cfg = cfg
        self.training_cfg = cfg["training_cfg"]
        self.task_to_id_dict = task_to_id_dict
        self.colourmaps = colourmaps
        self.task_z_code_dict = task_z_code_dict

        # Reverse task_to_id_dict to get id_to_task_dict
        id_to_task_dict = {}
        for k in task_to_id_dict:
            id_to_task_dict[task_to_id_dict[k]] = k
        self.id_to_task_dict = id_to_task_dict

        # Log all the configuration hyperparameters
        self.save_hyperparameters(cfg)

        # Create the model
        self._create_model()

        # Create losses
        self.losses = self._create_losses()

        # Create metrics
        self.metrics = self._create_metrics()
        
    def _one_step(self, batch, batch_idx, which_split):
        """
        Processes one batch
        """
        raise NotImplementedError    

    def _compute_loss(self, pred_logits, preds, batch):
        """
        Calculate losses for all considered tasks.
        Returns a dictionary with all loss components
        as well as the total loss.
        """
        raise NotImplementedError

    def _update_metrics_step_end(self, pred_logits, preds, batch):
        """
        Metric update which happens at every step
        """
        raise NotImplementedError
        
    def _get_optim_dict(self):
        """
        Returns a list of dictionaries.
        Each dictionary contains model parameters, their corresponding lr and wd.
        """
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        # Execute one forward pass
        # and computes the losses
        return self._one_step(
            batch=batch, 
            batch_idx=batch_idx, 
            which_split="train"
        )

    def training_step_end(self, out):
        # Log losses
        self._log_losses_step_end(
            loss_dict=out["loss_dict"], 
            which_split="train",
        )

        # Update metrics
        self._update_metrics_step_end(
            pred_logits=out["pred_logits"],
            preds=out["preds"],
            batch=out["batch"]
        )

        # Log metrics
        self._log_metrics_step_end(
            which_split="train",
        )
        
        return out["loss_dict"]["loss_total"]

    def training_epoch_end(self, outs):
        # Log and reset metrics
        self._log_and_reset_metrics_epoch(outs=outs, which_split="train")

    def validation_step(self, batch, batch_idx):
        # Execute one batch step
        return self._one_step(
            batch=batch, 
            batch_idx=batch_idx, 
            which_split="val"
        )
    
    def validation_step_end(self, out):
        # Log losses
        self._log_losses_step_end(
            loss_dict=out["loss_dict"], 
            which_split="val",
        )

        # Update metrics
        self._update_metrics_step_end(
            pred_logits=out["pred_logits"],
            preds=out["preds"],
            batch=out["batch"]
        )

        # Log metrics
        self._log_metrics_step_end(
            which_split="val",
        )
        
        return out["loss_dict"]["loss_total"]

    def validation_epoch_end(self, outs):
        # Log and reset metrics
        self._log_and_reset_metrics_epoch(outs=outs, which_split="val")

    # TODO No test split yet
    # TODO Implement this as a call to the validation steps?
    # def test_step(self, batch, batch_idx):
    #     pass
    # def test_step_end(self):
    #     pass
    # def test_epoch_end(self, outs):
    #     pass

    def _create_model(self):
        """
        Creates all of the components of the differentiable model.
        """

        # Construct a task_z code ditionary where the keys are task id's
        task_z_code_dict = self.task_z_code_dict#.copy()
        new_task_z_code_dict = {}
        for k in task_z_code_dict:
            new_k = self.task_to_id_dict[k]
            new_task_z_code_dict[new_k] = task_z_code_dict[k]  

        # Create the specified model
        if self.cfg["model_cfg"]["which_model"] == "composite_tasking_net_v1":
            model = CompositeTaskingNetV1(
                cfg=self.cfg["model_cfg"],
                task_z_code_dict=new_task_z_code_dict
            )
        elif self.cfg["model_cfg"]["which_model"] == "multi_head_net_v1":
            model = MultiHeadNetV1(
                cfg=self.cfg["model_cfg"],
                task_z_code_dict=new_task_z_code_dict
            )
        elif self.cfg["model_cfg"]["which_model"] == "multi_net_v1":
            model = MultiNetV1(
                cfg=self.cfg["model_cfg"],
                task_z_code_dict=new_task_z_code_dict
            )
        elif self.cfg["model_cfg"]["which_model"] == "single_tasking_net_v1":
            model = SingleTaskingNetV1(
                cfg=self.cfg["model_cfg"]
            )
        else:
            raise NotImplementedError

        self.model = model

    def configure_optimizers(self):
        """
        Creates a training optimizer , links it to the created model and returns it.
        If there is need for a very specific optimizer, 
        this method can be re-defined in the child class
        """

        # Returns a list of dictionaries
        # Each dictionary contains model parameters, their corresponding lr and wd.
        optim_list = self._get_optim_dict()

        # Which optimizer?
        if self.training_cfg["optimizer"].lower() == "sgd":
            for i in range(len(optim_list)):
                optim_list[i]["momentum"] = self.training_cfg["sgd_momentum"]
                optim_list[i]["nesterov"] = self.training_cfg["sgd_nesterov_momentum"]
            optimizer = torch.optim.SGD(optim_list)
        elif self.training_cfg["optimizer"].lower() == "adamw":
            optimizer = torch.optim.AdamW(optim_list)
        elif self.training_cfg["optimizer"].lower() == "adam":
            optimizer = torch.optim.Adam(optim_list)
        else:
            raise AssertionError

        # TODO Implement ReduceLROnPlateau
        # TODO (this will complicate things a little bit since it needs access to the loss)
        # Which learning rate scheduler to use?
        if self.training_cfg["lr_scheduler"].lower() == "step_lr":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, 
                step_size=self.training_cfg["lr_scheduler_step"], 
                gamma=self.training_cfg["lr_scheduler_factor"], 
                last_epoch=-1, 
            )
        elif self.training_cfg["lr_scheduler"].lower() == "cosine":
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                T_max=int(0.95*self.training_cfg["pl_max_epochs"]),
                eta_min=self.training_cfg["lr"]/1000.0,
                last_epoch=-1,
                verbose=False,
            )
        elif self.training_cfg["lr_scheduler"].lower() == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, 
                mode='min', 
                factor=self.training_cfg["lr_scheduler_factor"], 
                patience=self.training_cfg["lr_scheduler_patience"], 
            )

            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val/loss/loss_total",
            }
        else:
            raise NotImplementedError

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler
        }      

    def _manual_backprop(self, loss):
        """
        Implement backgpropagation manually.
        Torch-lightning does this automatically,
        but for some specific applications this is needed.
        """
        # Fetch the defined optimizer
        # TODO use_pl_optimizer??????
        optimizer = self.optimizers(use_pl_optimizer=False) 

        # Delete calculated gradients 
        # TODO Is this needed?
        optimizer.zero_grad()

        # Calculate the loss gradients
        self.manual_backward(loss)

        # Update network weights with calculated gradients  
        optimizer.step()

        # Delete calculated gradients
        optimizer.zero_grad()

    def _create_losses(self):
        losses = {}
        for task in self.cfg["data_set_cfg"]["task_list"]:
            if task == "normals":
                losses[task] = CosineSImilarityLoss(
                    normalize_vec=True,
                    zero_discard=True
                )
            elif task == "edges": 
                losses[task] = BalancedBCELoss(pos_weight=0.95)
            elif task == "saliency":
                losses[task] = BalancedBCELoss()
            elif task in ["seg", "parts"]:
                losses[task] = FocalCEGridToSeg(
                    cls_centroids=self.colourmaps[task],
                    focal_gamma=self.training_cfg["seg_l_focal_gamma"],
                    discard_empty=False,
                )
            else:
                raise NotImplementedError
        # nn.ModuleDict so the right device is propagated, etc
        return nn.ModuleDict(losses) 

    def _log_losses_step_end(self, loss_dict, which_split):
        """
        Loss logging which happens at every step
        """

        # TODO The TorchLightning will be accumulating the loss
        # TODO and calculating the mean loss in the end.
        # TODO Every task was probably not present for every batch element
        # TODO And thus to be very precice that would need to be taken into account
        # TODO But this is not a major issue, and only metrics need to be that precice
        for component_name in loss_dict:
            loss_component = loss_dict[component_name].detach()
            prog_bar = True if component_name == "loss_total" else False
            self.log(
                name=f"{which_split}/loss/{component_name}", 
                value=loss_component,
                on_step=True if which_split=="train" else False,
                on_epoch=True, 
                prog_bar=prog_bar, 
                logger=True
            )

    def _create_metrics(self):
        """
        Create the metric calculating functions for the correspondign tasks
        """
        metrics = {}
        for task in self.cfg["data_set_cfg"]["task_list"]:
            if task == "normals":
                metrics[task] = NormalsMetric(task=task)
            elif task in ["saliency", "edges"]:
                metrics[task] = EdgesSaliencyMetric(task=task)
            elif task in ["seg", "parts"]:
                metric_args = {
                    "cls_centroids": self.colourmaps[task],
                    "task": task,
                }
                if self.cfg["data_set_cfg"]["palette_mode"] in ["meaningful_labels_1", "meaningful_labels_3"]:
                    metric_args["cls_ids_of_interest"] = self.data_set["val"].used_seg_cls_ids
                metrics[task] = SegMetric(**metric_args)
            else:
                raise NotImplementedError
        
        # nn.ModuleDict so the right device is propagated, etc
        return nn.ModuleDict(metrics)

    def _log_metrics_step_end(self, which_split): 
        """
        Metric logging which happens at every step.
        For now, metrics are only calculated at the end of the epoch.
        """
        pass

    def _log_and_reset_metrics_epoch(self, outs, which_split):
        """
        Logging which happens at the end of the epoch
        """

        print(f"End of {which_split} epoch: {self.current_epoch}")
        print(f"-----------------------------------------------")

        for task in self.metrics:
            
            # Calculate the metric for the current task
            curr_metric = self.metrics[task].compute()
            # Restart the metric
            self.metrics[task].reset()
            
            # Go through each metric component
            for cur_met in curr_metric:
                if isinstance(curr_metric[cur_met], list):
                    # Optionally, more details from the metric 
                    # can be extracted anddisplayed
                    pass
                elif isinstance(curr_metric[cur_met], float):
                    # Log main componenets
                    self.log(
                        name=f"{which_split}/metric/{task}_{cur_met}_epoch", 
                        value=curr_metric[cur_met],
                        prog_bar=False,
                        logger=True
                    )
                    print(f"{which_split}/metric/{task}_{cur_met}_epoch:{curr_metric[cur_met]} ")
                else:
                    raise NotImplementedError
                        
            # Plot all different used learning rates for this epoch
            if which_split == "train":
                plotted_lr = []
                for j, param_group in enumerate(self.optimizers().param_groups):
                    if param_group['lr'] not in plotted_lr:
                        lr_curr = param_group['lr']
                        plotted_lr.append(lr_curr)
                        self.log(
                            name=f"misc/lr_{j}", 
                            value=lr_curr,
                            prog_bar=False,
                            logger=True
                        )
        print(f"\n\n")

    def _closure(self, batch):
        """
        This is a function which zeroes gradients, does a forward pass,
        calculates the loss and does a backwards pass.
        (No optimizer step!!!)
        This is required by some torch optimizers
        """

        # Fetch the defined optimizer
        optimizer = self.optimizers()

        # Zero out gradients
        optimizer.zero_grad()

        # Forward pass
        pred_logits, preds = self.forward(batch)

        # Calculating the loss
        loss_dict = self._compute_loss(
            pred_logits=pred_logits,
            preds=preds,
            batch=batch
        )

        # Do a backward pass
        loss_total = loss_dict["loss_total"]
        loss_total.backward()

        return loss_total
