from neuralforecast.auto import AutoNHITS, AutoLSTM , AutoNBEATS
import os
import time
import uuid
from datasetsforecast.losses import mse, mae, rmse
from datasetsforecast.evaluation import accuracy
from ray import tune
import pandas as pd
from neuralforecast import NeuralForecast
import time
from neuralforecast.losses.pytorch import MAE


config_nhits = {
    "input_size": tune.choice([5]),  # Length of input window
    "start_padding_enabled": True,
    "n_blocks": 5*[4],  # Number of blocks
    "mlp_units":  5*[[128, 128]],  # MLP units
    "n_pool_kernel_size": tune.choice([[2, 4, 8, 16],
                                       [4, 8, 16, 32],
                                       [8, 16, 32, 64]]),  # MaxPooling Kernel sizes
    "n_freq_downsample": tune.choice([[2, 4, 8, 16],
                                      [4, 8, 16, 32],
                                      [8, 16, 32, 64]]),  # Interpolation expressivity ratios
    "learning_rate": tune.loguniform(1e-5, 1e-2),  # Initial Learning rate
    "scaler_type": tune.choice([None]),  # Scaler type
    "max_steps": tune.choice([1500]),  # Max number of training iterations
    "batch_size": tune.choice([4, 8, 16]),  # Number of series in batch
    "windows_batch_size": tune.choice([256, 512, 1024]),  # Number of windows in batch
    "random_seed": tune.randint(1, 100),  # Random seed
}

config_lstm = {
    "input_size": tune.choice([5]),              # Length of input window
    "encoder_hidden_size": tune.choice([64, 128, 256]),            # Hidden size of LSTM cells
    "encoder_n_layers": tune.choice([2,4]),                   # Number of layers in LSTM
    "decoder_layers" : tune.choice([2,4]), 
    "decoder_hidden_size": tune.choice([64, 128, 256]),
    "learning_rate": tune.loguniform(1e-4, 1e-2),             # Initial Learning rate
    "scaler_type": tune.choice(['robust','standard']),                   # Scaler type
    "max_steps": tune.choice([1500]),                    # Max number of training iterations
    "batch_size": tune.choice([1, 4, 10]),                        # Number of series in batch
    "random_seed": tune.randint(1, 20),                       # Random seed
}
config_nbeats= {
    
    "input_size": tune.choice([10]),              # Length of input window
    "start_padding_enabled": True,
    "n_blocks": 5*[1],                                              # Length of input window
    "mlp_units": 5 * [[64, 64]],                                  # Length of input window
    "learning_rate": tune.loguniform(1e-4, 1e-2),                   # Initial Learning rate
    "scaler_type": tune.choice([None]),                             # Scaler type
    "max_steps": tune.choice([1000]),                               # Max number of training iterations
    "batch_size": tune.choice([1, 4, 10]),                          # Number of series in batch
    "windows_batch_size": tune.choice([128, 256, 512]),             # Number of windows in batch
    "random_seed": tune.randint(1, 20),   
    
}
nf = NeuralForecast(
    models=[
        AutoNHITS(h=1, config=config_nhits, loss=MAE(), num_samples=30),
        AutoLSTM(h=1, config=config_lstm, loss=MAE(), num_samples=30), 
        #AutoNBEATS (h=2, config=None, loss=MAE(),  num_samples=2),

    ],
    freq='H'
)

# Define the main directory where your container data directories are located
main_directory = '/home/razine.ghorab/projects/projectX/test_all/clustering/high_containers'  # Update with the correct path

# Get a list of subdirectories (each representing a container) within the main directory
container_directories = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
master_df = pd.DataFrame()
# Set a counter variable to control the number of iterations
counter_limit = 1
counter = 0

start_time = time.time()
# Process each container directory and its CSV file
for container_directory in container_directories:
    if counter >= counter_limit:
        
        break  # Exit the loop if the counter exceeds the limit
    
    #counter = counter + 1
    container_directory_path = os.path.join(main_directory, container_directory)

    # Find the CSV file inside the container directory
    csv_files = [f for f in os.listdir(container_directory_path) if f.endswith('original.csv')]
    print(csv_files[0])

    if not csv_files:
        print(f"No CSV files found in the directory: {container_directory_path}")
        continue

    container_file_path = os.path.join(container_directory_path, csv_files[0])

    data = pd.read_csv(container_file_path, skiprows=1, header=None, names=['y'])

    # Generate an integer index for ds
    data['ds'] = range(1, len(data) + 1)

    # Extract the dataset name from the file path to use as the unique_id
    dataset_name = os.path.splitext(os.path.basename(container_file_path))[0]
    data['unique_id'] = dataset_name  # Set the unique_id column to the dataset name

    # Rearrange columns to match the required format
    data = data[['unique_id', 'ds', 'y']]

    # Verify the transformation
    print(data.head())

    cv_df = nf.cross_validation(data, step_size=1,n_windows=100)
    evaluation_df = accuracy(cv_df, [mae], agg_by=['unique_id'])
    #evaluation_df.head()


    master_df = pd.concat([master_df, evaluation_df], ignore_index=True)

end_time = time.time()
execution_time = end_time - start_time

with open('modified_stepsize_execution_time.txt', 'w') as file:
    file.write(f"Execution Time: {execution_time} seconds")
master_df = master_df.drop('metric', axis=1)
master_df.to_csv('modified_stepsize_python_script_all_model_bench.csv', index=False)