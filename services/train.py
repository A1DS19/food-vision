
import torch
import torch.nn as nn
from timeit import default_timer as timer
from model_builder import BuildModel
from data_setup import SetupData
from utils import save_model
from engine import TrainTestStep
from pathlib import Path


def run():
    start_timer = timer()

    data_path = Path('./data/pizza_steak_sushi/')

    save_model_path = Path('./models/')
    train_dir = data_path / 'train'
    test_dir = data_path / 'test'
    save_model_dir = save_model_path
    model_name = 'test_model_v3.pth'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    setup_data = SetupData(train_dir=str(train_dir),
                           test_dir=str(test_dir))

    train_dataloader, test_dataloader, class_names = setup_data.create_dataloaders()

    input_shape = 3
    output_shape = len(class_names)
    hidden_units = 10

    build_model = BuildModel(device=device,
                             input_shape=input_shape,
                             output_shape=output_shape,
                             hidden_units=hidden_units)

    model = build_model.build_model()

    optimizer = torch.optim.Adam(params=model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    epochs = 5

    train_test_step = TrainTestStep(model=model,
                                    train_dataloader=train_dataloader,
                                    test_dataloader=test_dataloader,
                                    optimizer=optimizer,
                                    loss_fn=loss_fn,
                                    epochs=epochs,
                                    device=device)

    results = train_test_step.train_model()

    end_timer = timer()
    total_time = end_timer - start_timer

    print(f'[INFO] Model took {total_time:.2f} seconds to train.')

    save_model(model=model,
               target_dir=save_model_path,
               model_name=model_name)


if __name__ == '__main__':
    run()
