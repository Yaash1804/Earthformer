import torch
import torch.nn as nn
import lightning as L
from omegaconf import OmegaConf

# --- Assumed/Required Imports from Earthformer Repo ---
# This assumes the main LightningModule for Earthformer is defined in
# 'earthformer.lightning' and can be imported.
# Adjust the import path if your file structure is different.
try:
    from earthformer.lightning import EarthformerLitModule
except ImportError:
    print("WARNING: Could not import 'EarthformerLitModule' from 'earthformer.lightning'.")
    print("Please ensure the original Earthformer Lightning module is in the correct path.")
    # Define a placeholder if not found, to allow script to be read
    class EarthformerLitModule(L.LightningModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("ERROR: Using placeholder EarthformerLitModule!")
        def forward(selfself, x):
            raise NotImplementedError("Original EarthformerLitModule not found!")
# ----------------------------------------------------

from earthformer.models.convlstm import ConvLSTM


class DistillLitModule(L.LightningModule):
    """
    This LightningModule implements the requested Teacher-Student
    refinement architecture.

    It loads a pre-trained, frozen Earthformer model as the 'teacher'
    and a new ConvLSTM model as the 'student'.
    
    The training loop is defined as:
    1. Teacher(Input) -> Teacher_Prediction
    2. Student(Teacher_Prediction_T1) -> Student_Prediction_T1
    3. Loss(Student_Prediction_T1, Ground_Truth_T1)
    """

    def __init__(self, 
                 teacher_ckpt_path: str, 
                 teacher_config_path: str,
                 student_cfg: OmegaConf, 
                 optimizer_cfg: OmegaConf):
        """
        Initializes the system.
        
        Parameters
        ----------
        teacher_ckpt_path: str
            Path to the pre-trained Earthformer.ckpt file.
        teacher_config_path: str
            Path to the.yaml config file used to train the teacher.
        student_cfg: OmegaConf
            OmegaConf object containing configuration for the student (ConvLSTM).
            Expected keys: input_dim, hidden_dim (list), kernel_size, num_layers.
        optimizer_cfg: OmegaConf
            OmegaConf object containing configuration for the optimizer.
            Expected keys: lr, weight_decay.
        """
        super().__init__()
        # Save hyperparameters for logging and checkpointing
        self.save_hyperparameters(ignore=['teacher_ckpt_path', 'teacher_config_path'])

        self.student_cfg = student_cfg
        self.optimizer_cfg = optimizer_cfg

        # 1. Load the pre-trained, frozen teacher model
        self.teacher_model = self._load_teacher(teacher_ckpt_path, teacher_config_path)

        # 2. Instantiate the new student model
        # The input_dim for the student is the output_dim of the teacher.
        self.student_model = ConvLSTM(
            input_dim=self.student_cfg.input_dim,
            hidden_dim=list(self.student_cfg.hidden_dim),
            kernel_size=tuple(self.student_cfg.kernel_size),
            num_layers=self.student_cfg.num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False  # We only need the output of the last layer
        )

        # 3. Define the loss function
        # This will compare the student's output to the ground truth
        self.loss_fn = nn.MSELoss() # Or nn.L1Loss() if preferred

    def _load_teacher(self, ckpt_path: str, cfg_path: str) -> L.LightningModule:
        """
        Helper function to load the pre-trained Earthformer model
        from its checkpoint and config.
        """
        print(f"Loading teacher model configuration from: {cfg_path}")
        teacher_main_cfg = OmegaConf.load(cfg_path)
        
        # This assumes the model config is nested under a 'model' key,
        # which is standard in the amazon-science repo.
        teacher_model_cfg = teacher_main_cfg.model 

        print(f"Loading teacher model weights from: {ckpt_path}")
        # Use.load_from_checkpoint to properly instantiate the model
        # with its saved hyperparameters and load the weights [19, 20, 21]
        teacher_model = EarthformerLitModule.load_from_checkpoint(
            ckpt_path,
            cfg=teacher_model_cfg  # Pass the config to the model's __init__
        )
        
        # CRITICAL: Freeze the teacher model
        print("Freezing teacher model parameters...")
        teacher_model.freeze() # Sets.eval() and requires_grad=False [21]
        
        return teacher_model

    def forward(self, x: torch.Tensor):
        """
        Defines the full inference pass (Teacher -> Student).
        This is used by validation_step and predict_step.
        
        Parameters
        ----------
        x: torch.Tensor
            Input sequence, e.g., (B, 12, C_in, H, W)

        Returns
        -------
        torch.Tensor
            The student's final prediction, e.g., (B, 1, C_out, H, W)
        """
        # Run the teacher model in no_grad context
        with torch.no_grad():
            # teacher_model is an EarthformerLitModule, so calling it
            # executes its forward pass [4, 22]
            teacher_pred_seq = self.teacher_model(x)  # (B, 12_out, C_out, H, W)

            # Extract the first timestamp (T+1) for the student
            # Keep dim 1 for sequence length (T=1)
            student_input = teacher_pred_seq[:, 0:1, :, :, :] # (B, 1, C_out, H, W)
        
        # Run the student model (this part IS tracked by autograd)
        student_pred_list, _ = self.student_model(student_input)
        
        # student_pred_list is (B, 1, C_out, H, W)
        return student_pred_list

    def training_step(self, batch, batch_idx: int):
        """
        Implements the core training logic as requested.
        """
        x, y_true = batch  # x: (B, 12_in, C_in, H, W), y_true: (B, 12_out, C_out, H, W)

        # 1. & 2. Teacher Inference (Frozen)
        # Wrap in no_grad to ensure no gradients are computed for the teacher
        # and to save memory.
        with torch.no_grad():
            teacher_pred_seq = self.teacher_model(x) # (B, 12, C_out, H, W)
        
        # 3. & 4. Data Extraction and Student Input
        # Extract the T+1 prediction from the teacher
        student_input = teacher_pred_seq[:, 0:1, :, :, :] # (B, 1, C_out, H, W)

        # 5. Student Prediction (Trainable)
        # Gradients will flow from this step back to the student
        student_pred, _ = self.student_model(student_input) # (B, 1, C_out, H, W)

        # 6. Loss Calculation
        # Extract the T+1 ground truth frame to compare against
        ground_truth_t1 = y_true[:, 0:1, :, :, :] # (B, 1, C_out, H, W)

        loss = self.loss_fn(student_pred, ground_truth_t1)

        # 7. Logging
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        # 8. Backpropagation (handled by PyTorch Lightning)
        return loss

    def validation_step(self, batch, batch_idx: int):
        """
        Implements the validation logic.
        """
        x, y_true = batch # x: (B, 12_in, C_in, H, W), y_true: (B, 12_out, C_out, H, W)
        
        # Use the self.forward() method for the full T->S pass
        student_pred = self(x) # (B, 1, C_out, H, W)

        # Get T+1 ground truth
        ground_truth_t1 = y_true[:, 0:1, :, :, :] # (B, 1, C_out, H, W)

        val_loss = self.loss_fn(student_pred, ground_truth_t1)
        
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        """
        CRITICAL: Configures the optimizer to ONLY train the
        student_model parameters. The teacher_model parameters
        are frozen and will not be passed to the optimizer.
        """
        print("Configuring optimizer for STUDENT parameters only.")
        
        optimizer = torch.optim.Adam(
            self.student_model.parameters(), 
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay
        )
        
        # Optionally, add a learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        # return [optimizer], [scheduler]
        
        return optimizer
