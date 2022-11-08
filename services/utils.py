from pathlib import Path
import torch
import requests
import zipfile
from pathlib import Path


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
        data_url: str = 'https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip',
        data_folder: str = 'data',
        image_folder: str = 'pizza_steak_sushi') -> None:
    """Download dataset in zip format and unzip it.

    Keyword arguments:
    data_url -- url of zipped dataset
    data_folder -- name of folder where to store the data
    image_folder -- name of folder where to store the images
    Return: None
    """

    data_path = Path(f'{data_folder}/')
    image_path = data_path / image_folder

    if data_path.is_dir():
        print(f'{data_path} directory already exists')
    else:
        print(f'{data_path} directory does not exists, creating one...')
        data_path.mkdir(parents=True, exist_ok=True)

    if image_path.is_dir():
        print(f'{image_path} directory already exists')
    else:
        print(f'{image_path} directory does not exists, creating one...')
        image_path.mkdir(parents=True, exist_ok=True)

        with open(data_path / f'{image_folder}.zip', 'wb') as f:
            print('Downloading, dataset...')
            req = requests.get(data_url)
            f.write(req.content)
        print('Dataset downloaded')

        with zipfile.ZipFile(data_path / f'{image_folder}.zip', 'r') as zip_ref:
            print('Unzipping dataset...')
            zip_ref.extractall(image_path)
        print(f'Dataset unzipped in {image_path}')
