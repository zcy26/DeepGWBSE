import os
import logging
import time
import torch
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter 
from abc import ABC, abstractmethod
import copy 
import json
class Trainer(ABC):
    """
    Here we define a generic trainer for Sup- and Unsupervised learning.
        - `CHECKPOINT`: set to `True` to save the model each time a epoch finishes.
        - `BEST_MODEL`: set to `True` to save the model with the lowest loss in `model_name_best.pth`.
    """
    CHECKPOINT = True
    BEST_MODEL = True

    def __init__(self, model, optimizer, loss, 
                model_name="model", save_path=None, additional_metrics=None, scheduler=None) -> None:
        """
        The model will be saved each time a epoch finishes.
        In addition, the model with the lowest loss is saved in `model_name_best.pth`.

        Note that different subclasses are expected to put different requirements how `loss` is called.
        We do not impose hard constraints on the function signature of `loss`. 
        See :func:`get_loss`.
        """
        
        # Temporary files
        # Saving models
        if save_path == None:
            save_path = os.path.join(os.getcwd(), model_name + ".save")
        self.save_path = save_path
        try:
            os.mkdir(self.save_path)
        except FileNotFoundError:
            print("Parent directory not found.")
        except FileExistsError:
            print("Save path already exists. Working with the existing directory.")
        except Exception as e:
            print(e)
        self.model_name = model_name
        self.current_model_path = os.path.join(self.save_path, f"{self.model_name}.pth")
        self.best_model_path = os.path.join(self.save_path, f"{self.model_name}_best.pth")
        self.model_config_path = os.path.join(self.save_path, f"model_config.json")
        self.best_model = Trainer.BEST_MODEL
        self.checkpoint = Trainer.CHECKPOINT
        self.minimum_validation_loss = math.inf
        
        # Logging
        self.logger = logging.getLogger(f"logger of model {model_name}")
        self.logger.handlers.clear() # clear all existing handlers
        self.logger.setLevel(logging.DEBUG)
        self.verbose_logger = logging.getLogger(f"logger about everything of model {model_name}")
        self.verbose_logger.handlers.clear() # clear all existing handlers
        self.verbose_logger.setLevel(logging.DEBUG)
        # The log file
        self.train_log_path = os.path.join(self.save_path, "log.txt")
        file_handler = logging.FileHandler(self.train_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        # Everything going to self.logger will go to both the log file and the console;
        # everything going to self.verbose_logger will only go to the file
        self.logger.addHandler(file_handler)
        self.verbose_logger.addHandler(file_handler)
        # Print the log to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)
       
        # Move the model to GPU, if any
        self.loaded_from_file = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.logger.warn("The program is running on CPUs. Performance may be bad!")
        self.model = model.to(self.device)    
        self.initial_state = copy.deepcopy(self.model.state_dict()) # save the initial state of the model for training from scratch
        # export model_config to a json file
        if hasattr(model, "model_config"):
            self.model_config = model.model_config
            with open(self.model_config_path, "w") as f:
                json.dump(self.model_config, f, indent=4)
            self.logger.info(f"Model config is found and saved to {self.model_config_path}")
        else:
            self.model_config = None
            self.logger.info("Model config is not found. No model config is saved.")
        
        # Training data
        # Note that at initialization, by default we do not specify the datasets used in training:
        # they are to be specified when training actually happens,
        # and self.training_dataloader and self.validation_dataloader record the datasets used in the last training
        self.training_dataloader = None
        self.validation_dataloader = None
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss = loss
        self.additional_metrics = additional_metrics
        
        self.tb_writer = SummaryWriter(os.path.join(self.save_path, f"{self.model_name}-tensorboard"))

        pass
    
    @classmethod
    # @abstractmethod
    def load_all_from_exisiting_dir(cls, model_save:str):
        """
        Load the model, loss, and other info from an existing directory.
        You don't need to specify anything but the model name/path
        motivation: we won't remember specify parameter of the model for existing models and loss.
        """
        # TODO: make this a abstract method
        return cls

    @staticmethod
    def configure_model(model, model_config_path:str):
        """
        Configure the model from a json file.
        The json file should be in the format of model_config.json.
        """
        model_config_json_path = os.path.join(model_config_path, "model_config.json")
        assert os.path.exists(model_config_json_path), f"Model config file {model_config_json_path} does not exist."
        with open( model_config_json_path, "r") as f:
            model_config = json.load(f)
        return model(**model_config)

    def load_model(self, load_best=False):
        """
        load the model from the file.
        load_best:
            True: load the best model for evaluation
            False: load the current model for continued training
        """
        if load_best:
            if os.path.exists(self.best_model_path):
                self.logger.info("Best model loaded.")
                self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device), strict=False)
                self.loaded_from_file = True
            else:
                self.logger.info("Best model does not exist.")
                self.loaded_from_file = False
        
        else:
            if os.path.exists(self.current_model_path):
                self.logger.info("Current model loaded.")
                self.model.load_state_dict(torch.load(self.current_model_path, map_location=self.device), strict=False)
                self.loaded_from_file = True
            else:
                self.logger.info("Current model does not exist.")
                self.loaded_from_file = False



    def train_each_epoch(self, epoch_idx: int, training_dataloader, validation_dataloader):
        """
        What is presented here is a generic training procedure.
        The method can be overriden by another procedure in subclasses.
        In this case, do not forget to call `self.record` at the end of each epoch.
        """
        self.model.train()
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)

        total_loss = 0.0
        
        # start.record()
        for x in tqdm(training_dataloader, f"Epoch {epoch_idx+1}"):
            self.optimizer.zero_grad()
            # We directly feed the output of the dataloader to get_loss:
            # self.get_loss has the responsibility to properly handle the structure of x!
            this_loss = self.get_loss(x)
            this_loss.backward()
        
            # add clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)

            self.optimizer.step()
            total_loss += this_loss.item()
        if self.scheduler is not None:
            self.scheduler.step()
        
        if self.additional_metrics is not None:
            validation_loss, additional_metrics_info =  self.validate(validation_dataloader, get_additional_loss=self.get_additional_loss)
        else:
            validation_loss, additional_metrics_info =  self.validate(validation_dataloader)

        # torch.cuda.synchronize()  # Ensure all kernels are finished
        # end.record()

        self.record(epoch_idx, 
                    training_loss=total_loss / len(training_dataloader),
                    validation_loss=validation_loss,
                    elapsed_time=1,
                    additional_metrics_info=additional_metrics_info)
    
    @abstractmethod
    def get_loss(self, x):
        """
        This method uses `self.loss` to calculate the actual loss of one batch.
        Subclasses should override the definition of this method.
        We note that `x` is expected to be the output of a PyTorch dataloader:
        this means (a) it is a tuple containing two or more (or sometimes just one) tensor, and is not itself a tensor, and (b) it is likely stored in the host, not on the GPUs.
        """
        pass

    def get_additional_loss(self)->float:
        """
        This function must be overwritten if additional_metics is not None.
        Implementation Instruction:
            - run get_loss() to get self.prediction and self.target.
            - use self.additional_metrics(self.prediction, self.target) to get the additional loss.
            - return the additional loss.
            - get_additional_loss will only be called after get_loss() is called for validation: see self.validate()
            see bsetrainer.py for more details.
        """
        pass

    @torch.no_grad()
    def validate(self, dataloader=None, get_additional_loss=None):
        """
        The current implementation is to
        calculate the loss of the current model on a validation dataset.
        `get_additional_loss` allows the user to define a different loss function for validation.
        The default loss function is the same as the one used in training.
        """
        self.model.eval()
        
        total_loss_additional = 0.0 if get_additional_loss is not None else ""
        total_loss = 0.0
        if dataloader is None:
            assert self.validation_dataloader is not None, "A validation dataloader has to be passed"
            dataloader = self.validation_dataloader
        for x in dataloader:
            this_loss = self.get_loss(x)
            total_loss += this_loss.item()
            if get_additional_loss is not None:
                total_loss_additional +=get_additional_loss() 

        total_loss_additional = total_loss_additional/len(dataloader) if get_additional_loss is not None else ""
        total_loss = total_loss / len(dataloader)

        return total_loss, total_loss_additional

    def train(self, epoches: int, training_dataloader, validation_dataloader, continued=False):
        """
        The batch size should already be defined in `optimizer`.
        In this method we do not provide hooks for defining the batch size.
        """
        #self.training_dataset = ...
        print("Continued training:", continued, "\nLoaded from file:", self.loaded_from_file)
        if self.loaded_from_file and not continued:
            # self.logger.warn("Model loaded from file: no training is done. Set continued to True to train on top of existing model.")
            self.logger.warning("Loaded model is detected! But continued is False => Training from scratch.")
            self.model.load_state_dict(self.initial_state)
            # return
        elif not self.loaded_from_file and continued:
            self.logger.warning("Loaded model is not detected! => Training from scratch.")
        
        elif self.loaded_from_file and continued:
            self.logger.info("Training from the loaded model.")

        elif not self.loaded_from_file and not continued:
            self.logger.info("Training from scratch.")
            self.model.load_state_dict(self.initial_state)
 
        for epoch in range(epoches):
            self.model.train()
            self.train_each_epoch(epoch, training_dataloader, validation_dataloader)
            self.loaded_from_file = True
            if self.checkpoint:
                torch.save(self.model.state_dict(), self.current_model_path)
        
        torch.save(self.model.state_dict(), self.current_model_path)
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.verbose_logger.info("The final model saved. Training ends.")

    @abstractmethod
    def evaluate(self, input=None, **kwargs):
        """
        To be overwritten by subclasses; you decide what output to return.
        This method takes an optional `input` and return the predication of the model based on `input`.
        When `input` is not given, its default value is the first batch in the validation dataset.
        The return values should better be NumPy arrays,
        instead of tensors or tensors on GPUs:
        when the latter is needed, you can always directly call `self.model(...)`.
        """
        #self.model.eval()
        # ...
        pass



    def record(self, epoch: int, **kwargs):
        """
        This method is expected to be called by `train_each_epoch`.
        It should be called once an epoch finishes.
        `train_each_epoch` is expected to pass information like loss to this method.
        Details about how this information is collected are left to subclasses to implement.
        """
 
        training_loss = kwargs["training_loss"]
        validation_loss = kwargs["validation_loss"]
        learning_rate = self.optimizer.param_groups[0]["lr"]
        elapsed_time = kwargs["elapsed_time"]
        additional_metrics_info = kwargs["additional_metrics_info"]
        additional_metrics_info = f'{additional_metrics_info:.2e}' if additional_metrics_info != "" else ""

        self.tb_writer.add_scalar("Training loss", training_loss, global_step=epoch)
        self.tb_writer.add_scalar("Validation loss", validation_loss, global_step=epoch)

        if self.minimum_validation_loss > validation_loss:
            torch.save(self.model.state_dict(), self.best_model_path)
            self.minimum_validation_loss = validation_loss
            self.verbose_logger.info(f"Eopch {epoch+1} | train. loss {training_loss:.2e} | val. loss {validation_loss:.2e} | val. metrics: {additional_metrics_info}| learning rate: {learning_rate:.2e} | (Best model)")
        else:
            self.verbose_logger.info(f"Eopch {epoch+1} | train. loss {training_loss:.2e} | val. loss {validation_loss:.2e} | val. metrics: {additional_metrics_info}| learning rate: {learning_rate:.2e} |")
        


if __name__ == "__main__":
    """
    Four scenarios:
    1. Training from scratch
    2. Training from a checkpoint (last saved or best model)
    3. Training from scratch, pause and evaluate the model, then continue training
    4. Training from scratch, pause and evaluate the model, training from scratch again
    """

    # Scenario 1: Training from scratch
    trainer = Trainer(...)
    trainer.train(...)

    # Scenario 2: Training from a checkpoint
    trainer = Trainer(...)
    trainer.load_model(load_best=False)
    trainer.train(continued=True) 

    # Scenario 3: Training from scratch, pause and evaluate the model, then continue training
    trainer = Trainer(...)
    trainer.train(...)
    trainer.load_model(load_best=True)
    trainer.evaluate(...)
    trainer.load_model(load_best=False)
    trainer.train(continued=True) 

    # Scenario 4: Training from scratch, pause and evaluate the model, training from scratch again
    trainer = Trainer(...)
    trainer.train(...)
    trainer.load_model(load_best=True)
    trainer.evaluate(...)
    trainer.train(continued=False)

    # Configure a model
    model = Trainer.configure_model(model=..., model_config_path=...)