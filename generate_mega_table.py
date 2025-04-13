import matplotlib.pyplot as plt
from pathlib import Path
import json
import pickle
from datetime import datetime
import pandas as pd
import pUtil

script_dir = Path(__file__).resolve().parent

# This function is used to extract the numbers from the dataset folder name for sorting so our table is in order.
def extract_numbers(folder_name):
    parts = folder_name.split('_') 
    numbers = [int(part) for part in parts if part.isdigit()]
    return tuple(numbers)

# Conventionally, `dataset` is used for the dataset, i.e. dataset_5_2, and `dataset_out_dir` is used for the output directory, i.e. dataset_5_2_1.
# Here, I am differentiating between the two by using `dataset_name` for the output directory, so `dataset` can retain the original meaning.

trained_models_path = script_dir / 'trained_models'
generated_samples_path = script_dir / 'generated_samples'
dataset_names_in_trained_models = [folder.name for folder in trained_models_path.iterdir() if folder.is_dir()]
dataset_names_in_generated_samples = [folder.name for folder in generated_samples_path.iterdir() if folder.is_dir()]
# Take the intersection of the two lists
dataset_names = [folder_name for folder_name in dataset_names_in_trained_models if folder_name in dataset_names_in_generated_samples]
dataset_names = sorted(dataset_names, key=extract_numbers)

collected_information = {}

# We assume dataset is the name of the dataset_out_dir (i.e. dataset_5_2_1, not dataset_5_2).
for dataset_name in dataset_names:
    # -------------------------------------------------------------------------------
    # First, we retrieve training information from the training log file.
    # We are looking for:
    # 1) The best train_loss and val_loss and the corresponding iterations and epoch
    # 2) Training time, trained iterations, and trained epochs
    # -------------------------------------------------------------------------------
    
    # Get training information from the training log file
    training_log_file = str(next(Path(pUtil.get_training_dir(dataset_name)).glob("*.jsonl"), None))
    dataset = None
    start_training_time = None
    end_training_time = None
    # Since we only test and save checkpoint every eval_interval, we need to keep track of checkpoint data separately for saved train_loss and val_loss
    train_data_dict = {}
    checkpoint_data_dict = {}
    with open(training_log_file, 'r') as data_file:
        for line in data_file:
            jdata = json.loads(line)
            if jdata['message'] == "Training progress" and 'iter' in jdata:
                train_data_dict[jdata['iter']] = (jdata['train_loss'], jdata['val_loss'])
                # Starting training time is when iteration is 0
                if jdata['iter'] == 0:
                    start_training_time = datetime.fromisoformat(jdata['timestamp'])
            elif jdata['message'] == "Training progress: checking checkpoint conditions":
                checkpoint_data_dict[jdata['step']] = (datetime.fromisoformat(jdata['timestamp']), jdata['train_loss'], jdata['val_loss'])
            elif jdata['message'] == "Training configuration":
                dataset = jdata['dataset']

    # This is for all data
    sorted_iterations = sorted(train_data_dict.keys())
    train_loss = [train_data_dict[i][0] for i in sorted_iterations]
    val_loss = [train_data_dict[i][1] for i in sorted_iterations]

    # This is for the checkpointed data
    min_step, (min_timestamp, min_saved_train_loss, min_saved_val_loss) = min(checkpoint_data_dict.items(), key=lambda item: item[1][2])
    trained_iters = min_step
    
    training_time_seconds = abs(min_timestamp - start_training_time).total_seconds()
    
    # -------------------------------------------------------------------------------
    # Next, we retrieve the training information from the config file.
    # -------------------------------------------------------------------------------
    
    config_file = Path('config', f'{dataset_name}.json')
    with open(config_file) as f:
        config = json.load(f)
        training_config = config.get('training_config', {})
        batch_size = training_config.get('batch_size', -1)
        block_size = training_config.get('block_size', -1)
        context_events = training_config.get('context_particles', -1)
        learning_rate = training_config.get('learning_rate', -1)
        min_learning_rate = training_config.get('min_lr', -1)
        
    # -------------------------------------------------------------------------------
    # Next, we retrieve the dataset information from the meta file.
    # -------------------------------------------------------------------------------
    
    # dataset is retrieved from the training log file since that is the most reliable way to do so in case
    # we ever stop following our naming scheme.
    meta_file = Path('data', dataset, 'outputs', 'meta.pkl')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f)
        vocab_size = meta['vocab_size']
        total_events = meta['total_events']
        num_train_events = meta['num_train_events']
        num_train_tokens = meta['num_train_tokens']
        num_val_events = meta['num_val_events']
        num_val_tokens = meta['num_val_tokens']
        max_sequence_length = meta['max_sequence_length']
    
    # -------------------------------------------------------------------------------
    # Next, we retrieve metric information from the relevant generated_samples folder.
    # -------------------------------------------------------------------------------
    
    latest_sampling_dir = pUtil.get_latest_sampling_dir(dataset_name)
    metrics_dump = latest_sampling_dir / 'metrics_results.json'
    with open(metrics_dump, "r") as f:
        metrics = json.load(f)
        metrics_coverage = metrics['coverage']
        metrics_mmd = metrics['mmd']
        metrics_kpd_median = metrics['kpd_median']
        metrics_kpd_error = metrics['kpd_error']
        metrics_fpd_value = metrics["fpd_value"]
        metrics_fpd_error = metrics["fpd_error"]
    
    # -------------------------------------------------------------------------------
    # Finally, we update collected_information.
    # -------------------------------------------------------------------------------
    
    # Calculate epochs trained
    iterations_per_epoch = num_train_tokens // (block_size * batch_size)
    epochs_trained = trained_iters / iterations_per_epoch
    
    # If we are using context_particles, calculate block size from that.
    if block_size == -1:
        block_size = context_events * max_sequence_length
    
    collected_information[dataset_name] = {
        # Dataset information
        'vocab_size': vocab_size,
        'total_events': total_events,
        'num_train_tokens': num_train_tokens,
        'num_val_tokens': num_val_tokens,
        'num_train_events': num_train_events,
        'num_val_events': num_val_events,
        # Model information
        'batch_size': batch_size,
        'block_size': block_size,
        'learning_rate': learning_rate,
        'min_learning_rate': min_learning_rate,
        # Training results
        'training_time_seconds': training_time_seconds,
        'best_train_loss': min_saved_train_loss,
        'best_val_loss': min_saved_val_loss,
        'epochs_trained': epochs_trained,
        'trained_iterations': trained_iters,
        # Metrics
        'coverage': metrics_coverage,
        'mmd': metrics_mmd,
        'kpd_median': metrics_kpd_median,
        'kpd_error': metrics_kpd_error,
        'fpd_value': metrics_fpd_value,
        'fpd_error': metrics_fpd_error,
    }

# Output collected_information as a .csv file

csv_filename = Path(script_dir, 'mega_table.csv')
with open(csv_filename, 'w') as out_file:
    first_key = next(iter(collected_information))
    param_names = ['dataset'] + [key for key in collected_information[first_key].keys()]
    header = ",".join(param_names)
    
    out_file.write(f'{header}\n')
    for dataset_name, info in collected_information.items():
        values_ls = []
        for value in info.values():
            values_ls.append(value)
        values = ",".join([str(value) for value in values_ls])
        out_file.write(f'{dataset_name},{values}\n')
        
# Output collected_information as a table in a .png file

def convert_seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

df = pd.read_csv(csv_filename)
# Some formatting work to ensure output is nice
rounding_columns = ['best_val_loss', 'best_train_loss', 'epochs_trained', 'coverage']
df[rounding_columns] = df[rounding_columns].map(lambda x: f"{x:,.5f}")
sci_notation_columns = ['mmd', 'kpd_median', 'kpd_error', 'fpd_value', 'fpd_error']
df[sci_notation_columns] = df[sci_notation_columns].map(lambda x: f"{x:.5E}")
comma_columns = ['vocab_size','total_events', 'num_train_tokens', 'num_val_tokens', 'num_train_events', 'num_val_events', 'batch_size', 'block_size', 'trained_iterations']
df[comma_columns] = df[comma_columns].map(lambda x: f"{int(x):,}")
df[['learning_rate', 'min_learning_rate']] = df[['learning_rate', 'min_learning_rate']].map(lambda x: f"{x:.10f}".rstrip('0').rstrip('.'))
df['training_time_seconds'] = df['training_time_seconds'].apply(convert_seconds_to_hms)

colors = {
    'dataset': '#FFDDC1',
    'model': '#C1E1FF',
    'training': '#D3FFCE',
    'metrics': '#FFB3B3'
}

column_groups = {
    'dataset': ['vocab_size', 'total_events', 'num_train_tokens', 'num_val_tokens', 'num_train_events', 'num_val_events'],
    'model': ['batch_size', 'block_size', 'learning_rate', 'min_learning_rate'],
    'training': ['training_time_seconds', 'best_train_loss', 'best_val_loss', 'epochs_trained', 'trained_iterations'],
    'metrics': ['coverage', 'mmd', 'kpd_median', 'kpd_error', 'fpd_value', 'fpd_error']
}

fig, ax = plt.subplots(figsize=(12, 1))
ax.axis('off')

table = ax.table(cellText=df.T.values, rowLabels=df.columns, cellLoc='right')

# Apply color-coding based on column groups
for i, col in enumerate(df.columns):
    for group, group_columns in column_groups.items():
        if col in group_columns:
            for j in range(len(df) + 1):
                table[(i, j - 1)].set_facecolor(colors[group])
       
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(4, 1.5)
plt.savefig('mega_table.png', bbox_inches='tight', dpi=300) 