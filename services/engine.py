
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch


class TrainTestStep():
    """
  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").
  """

    def __init__(self,
                 model: torch.nn.Module,
                 train_dataloader: torch.utils.data.DataLoader,
                 test_dataloader: torch.utils.data.DataLoader,
                 optimizer: torch.optim.Optimizer,
                 loss_fn: torch.nn.Module,
                 epochs: int,
                 device: torch.device,
                 tensorboard_log_dir: str = None):
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.device = device
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir)

    def train_step(self) -> Tuple[float, float]:
        """Trains a PyTorch model for a single epoch.

        Turns a target PyTorch model to training mode and then
        runs through all of the required training steps (forward
        pass, loss calculation, optimizer step).

        Returns:
          A tuple of training loss and training accuracy metrics. 
          In the form (train_loss, train_accuracy). For example:

          (0.1112, 0.8743)
      """
        self.model.train()
        train_loss, train_acc = 0, 0
        for batch, (X, y) in enumerate(self.train_dataloader):
            X, y = X.to(self.device), y.to(self.device)
            y_pred = self.model(X)
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        train_loss /= len(self.train_dataloader)
        train_acc /= len(self.train_dataloader)

        return train_loss, train_acc

    def test_step(self):
        """Tests a PyTorch model for a single epoch.

        Turns a target PyTorch model to "eval" mode and then performs
        a forward pass on a testing dataset.

        Returns:
         A tuple of testing loss and testing accuracy metrics.
         In the form (test_loss, test_accuracy). For example:

          (0.0223, 0.8985)
        """
        self.model.eval()
        test_loss, test_acc = 0, 0
        with torch.inference_mode():
            for batch, (X_test, y_test) in enumerate(self.test_dataloader):
                X_test, y_test = X_test.to(self.device), y_test.to(self.device)
                y_test_logits = self.model(X_test)
                y_test_loss = self.loss_fn(y_test_logits, y_test)
                test_loss += y_test_loss.item()
                test_pred_labels = y_test_logits.argmax(dim=1)
                test_acc += ((test_pred_labels ==
                             y_test).sum().item()/len(test_pred_labels))

            test_loss /= len(self.test_dataloader)
            test_acc /= len(self.test_dataloader)

            return test_loss, test_acc

    def train_model(self):
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []}

        for epoch in tqdm(range(self.epochs)):
            train_loss, train_acc = self.train_step()
            test_loss, test_acc = self.test_step()

            print(f"Epoch: {epoch+1} | "
                  f"train_loss: {train_loss:.4f} | "
                  f"train_acc: {train_acc:.4f} | "
                  f"test_loss: {test_loss:.4f} | "
                  f"test_acc: {test_acc:.4f}")

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            self.writer.add_scalars(main_tag='Loss', tag_scalar_dict={'train_loss': train_loss,
                                                                      'test_loss': test_loss},
                                    global_step=epoch)

            self.writer.add_scalars(main_tag='Accuracy', tag_scalar_dict={'train_loss': train_acc,
                                                                          'test_loss': test_acc},
                                    global_step=epoch)

            self.writer.add_graph(model=self.model, input_to_model=torch.randn(
                32, 3, 224, 224).to(self.device))

        self.writer.close()

        return results
