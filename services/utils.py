from pathlib import Path
import torch
import torch.nn as nn
import requests
import zipfile
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import glob
import random
import re
from torchvision.io import read_image
import torchvision
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict


def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
    """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                          exist_ok=True)

  # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"), "model_name should end with '.pt' or '.pth'"
    model_save_path = target_dir_path / model_name

  # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(),
               f=model_save_path)


def download_data(
        data_url: str = None,
        data_folder: str = 'data',
        image_folder: str = None) -> None:
    """Download dataset in zip format and unzip it.

    Keyword arguments:
    data_url -- url of zipped dataset
    data_folder -- name of folder where to store the data
    image_folder -- name of folder where to store the images
    Return: None
    """

    if data_url is None:
        raise Exception('data_url cannot be empty')
    if image_folder is None:
        raise Exception('image_folder cannot be empty')

    data_path = Path(f'{data_folder}')
    image_path = data_path / image_folder
    file_path = data_path / f'{image_folder}.zip'

    if data_path.is_dir():
        print(f'{data_path} directory already exists')
    else:
        print(f'{data_path} directory does not exists, creating one...')
        data_path.mkdir(parents=True, exist_ok=True)

    if image_path.is_dir():
        print(f'{image_path} directory already exists')
        print(f'dataset will not be downloaded')
    else:
        print(f'{image_path} directory does not exists, creating one...')
        image_path.mkdir(parents=True, exist_ok=True)

        print('Downloading, dataset...')
        req = requests.get(data_url)

        with open(file_path, 'wb') as f:
            f.write(req.content)
            print(f'Dataset downloaded in {file_path}')

        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            print('Unzipping dataset...')
            zip_ref.extractall(image_path)
            print(f'Dataset unzipped in {file_path}')


def plot_loss_curves(results):
    """Plots training curves of a results dictionary.
    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    loss = results["train_loss"]
    test_loss = results["test_loss"]

    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def pred_and_plot_images(transform: nn.Module,
                         model: nn.Module,
                         class_names: list,
                         val_dir: str,
                         device: str) -> None:
    """Predict and plot 10 random images on test dataset

    Keyword arguments:
    transform -- image transform
    model -- trained model
    class_names -- classes of data
    val_dir -- validation directory containing images
    device -- device (cpu, cuda)
    Return: None
    """

    model.to(device)
    model.eval()

    img_files = glob.glob(f'{str(val_dir)}/*/*.jpg')
    random_idxs = random.sample(range(len(img_files)), k=10)

    plt.figure(figsize=(30, 20))
    for idx, img_idx in enumerate(random_idxs):
        img_file = img_files[img_idx]
        img_file_short = img_file[22:]
        true_class = [re.search(class_name, img_file_short)
                      for class_name in class_names]
        true_class = [
            class_name for class_name in true_class if class_name is not None][0].group(0)
        raw_img = read_image(img_file).type(torch.float32)
        normalized_img = raw_img / 255.
        img = transform(normalized_img)

        with torch.inference_mode():
            img = img.unsqueeze(0).to(device)
            pred_logits = model(img)
            pred_probs = torch.softmax(pred_logits, dim=1)
            pred_label = torch.argmax(pred_probs)

        pred_class = class_names[pred_label]

        plt.subplot(5, 5, idx + 1)
        plt.imshow(normalized_img.squeeze(0).permute(1, 2, 0))
        title_msg = f'true class: {true_class} | predicted class {pred_class} {pred_probs.max():.3f}%'

        if true_class == pred_class:
            plt.title(title_msg, color='green')
        else:
            plt.title(title_msg, color='red')

        plt.axis(False)


def pred_and_plot_image(
    model: torch.nn.Module,
    image_path: str,
    class_names: list = None,
    transform=None,
    device: torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Makes a prediction on a target image with a trained model and plots the image.
    Args:
        model (torch.nn.Module): trained PyTorch image classification model.
        image_path (str): filepath to target image.
        class_names (List[str], optional): different class names for target image. Defaults to None.
        transform (_type_, optional): transform of target image. Defaults to None.
        device (torch.device, optional): target device to compute on. Defaults to "cuda" if torch.cuda.is_available() else "cpu".

    Returns:
        Matplotlib plot of target image and model prediction as title.
    """

    raw_image = torchvision.io.read_image(
        str(image_path))
    float_image = raw_image.type(torch.float32)
    target_image = raw_image / 255.0
    target_image = transform(target_image)

    model.to(device)
    model.eval()

    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        target_image_pred = model(target_image.to(device))

    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs)

    plt.imshow(raw_image.permute(1, 2, 0))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)


def create_summary_writer(
        experiment_name: str,
        extra: str,
        model_name):
    """Create tensorboard summary writer instance

    Keyword arguments:
    experiment_name -- name of the current experiment
    extra -- extra naming
    model_name -- name of the model

    Example:
    writer = create_summary_writer(
                experiment_name='data_10_percent',
                extra='50_epochs',
                model_name='efficientnet_b0')

    Return: summary writer instance
    """
    timestamp = datetime.now().strftime("%Y-%m-%d")
    if extra:
        log_dir = os.path.join(
            'runs', timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join('runs', timestamp, experiment_name, model_name)

    print(f'[INFO] Created SummaryWriter saving to {log_dir}')
    writer = SummaryWriter(log_dir=log_dir)
    return writer


def size_of_model_file(file_name: str):
    """Determine the size of a file based on it's path

    Keyword arguments:
     file_name --- path of file in str format.

    Return: None
    """
    file_dir = Path(file_name)

    if file_dir.is_file() == False:
        raise Exception('File does not exists')

    file_stats = os.stat(file_name)
    file_size = file_stats.st_size / (1024 * 1024)

    print(f'File size in megabytes is {file_size:.2f} MB')

    return file_size


def model_total_params(model: torch.nn.Module) -> int:
    """Get total parameters of model

    Keyword arguments:
    model -- Model from where to count the total number of parameters

    Return: Total number of parameters
    """
    return sum(torch.numel(param) for param in model.parameters())


def pred_and_time(model: nn.Module,
                  transform: nn.Module,
                  directory: str,
                  loss_fn=loss_fn,
                  class_names: List[str] = class_names,
                  device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                  print_realtime_inference: bool = False) -> List[Dict]:
    """Predict, time and get helpfull info to list of dictionaries
       containing data from prediction on images.

    Keyword arguments:
    model -- model to which make predictions
    transform -- list of transforms to apply to images
    directory -- path to where the images are located
    loss_fn -- loss function instance
    class_names -- list of classes of images
    device -- device in which to make predicions (cpu/cuda)
    print_realtime_inference -- print data of current inference

    Return: List of dictionaries containing info of predictions
    """

    selected_dir = Path(directory)

    if selected_dir.is_dir() == False:
        raise Exception('Path is not a directory')

    img_paths = glob.glob(f'{selected_dir}/*/*.jpg')
    pred_list = []

    model.eval()
    model = model.to(device)

    for image_path in tqdm(img_paths):
        image_path = Path(image_path)
        pred_dict = {}
        pred_dict['img_path'] = image_path
        true_class = image_path.parent.stem
        pred_dict['true_label'] = true_class

        start_inference = timer()
        image = torchvision.io.read_image(str(image_path))
        image = image.unsqueeze(dim=0)
        image = transform(image)
        image = image.to(device)

        with torch.inference_mode():
            logits = model(image)
            preds = torch.softmax(logits, dim=1)
            label = torch.argmax(preds)
            predicted_class = class_names[label.cpu()]

            pred_dict['pred_prob'] = round(preds.unsqueeze(0).max().item(), 4)
            pred_dict['pred_label'] = predicted_class
            pred_dict['successfull_pred'] = predicted_class == true_class

            stop_inference = timer()

        total_inference = round(stop_inference - start_inference, 4)
        pred_dict['total_inference_time'] = total_inference
        pred_list.append(pred_dict)

        if print_realtime_inference:
            print(f'{image_path} took {total_inference:.3f} seconds | predicted class {predicted_class} | true class {true_class}')

    inference_times = [item['total_inference_time'] for item in pred_list]
    mean_inference_time = torch.mean(torch.tensor(inference_times)).item()

    print(
        f'Mean inference time for {len(img_paths)} images on model {model.__class__.__name__} was {mean_inference_time} seconds')
    return pred_list, mean_inference_time
