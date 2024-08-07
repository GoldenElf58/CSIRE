import random
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ale_py import ALEInterface, roms

from utils import load, save_state, find_all_files, load_specific_state, mean_squared_error, plot_histogram


class Autoencoder(nn.Module):
    def __init__(self, input_dim, code_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, code_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Assuming input is normalized to [0, 1]
        )

    def forward(self, x):
        code = self.encoder(x)
        reconstruction = self.decoder(code)
        return code, reconstruction


def train_autoencoder(autoencoder: Autoencoder, data_loader, num_epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    results = []
    stop_event = threading.Event()
    current_iteration = [0]
    t0 = time.time()

    # Start the loading sign in a separate thread
    loader_thread = threading.Thread(target=load, args=(stop_event, num_epochs, current_iteration, results, t0))
    loader_thread.start()

    for epoch in range(num_epochs):
        for data in data_loader:
            inputs = data[0]
            optimizer.zero_grad()
            code, reconstruction = autoencoder(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
        results.append((epoch, epoch))
        current_iteration[0] += 1
    stop_event.set()
    print()


def collect_ram_state(ale):
    if ale.game_over():
        ale.reset_game()

    ram = ale.getRAM()
    ram_state = (np.array(ram, dtype=np.float32))

    return ram_state


def collect_ram_states(ale, rom_path, num_samples=1000):
    ale.loadROM(rom_path)
    ram_states = []
    files = find_all_files(base_filename='save-state')

    for file in files:
        ale.restoreState(load_specific_state(file))
        for i in range(int(num_samples / len(files))):
            # Take a random action
            action = ale.getLegalActionSet()[np.random.choice(len(ale.getLegalActionSet()))]
            ale.act(action)
            ram_states.append(collect_ram_state(ale))

    while len(ram_states) < num_samples:
        # Take a random action
        action = ale.getLegalActionSet()[np.random.choice(len(ale.getLegalActionSet()))]
        ale.act(action)
        ram_states.append(collect_ram_state(ale))

    # Normalize RAM states
    ram_states = np.array(ram_states) / 255.0
    ram_states = torch.FloatTensor(ram_states)
    return ram_states


def main():
    ale = ALEInterface()
    rom_path = roms.MontezumaRevenge
    ram_states = collect_ram_states(ale, rom_path, num_samples=15_000)
    print(ram_states)
    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(ram_states, batch_size=32, shuffle=True)

    autoencoder = Autoencoder(input_dim=128, code_dim=32)  # For Atari 2600 RAM states
    print("Ram States Collected")
    train_autoencoder(autoencoder, data_loader, num_epochs=100)
    save_state(autoencoder, 'autoencoder')
    # autoencoder = load_latest_state('autoencoder')
    mses = []
    ram_state_index = random.randint(0, 9_999)
    for i in range(10_000):
        torch.set_printoptions(sci_mode=False, precision=4)
        # autoencoder = load_latest_state(base_filename='autoencoder')
        code, reconstruction = autoencoder(ram_states[ram_state_index])
        reconstruction = reconstruction.tolist()
        mses.append(mean_squared_error(reconstruction, ram_states[i].tolist()))
    # print(sorted(mses))
    plot_histogram(mses)
    # error = elementwise_difference(reconstruction, ram_states[ram_state_index].tolist())


if __name__ == "__main__":
    print("Program Started")
    main()
