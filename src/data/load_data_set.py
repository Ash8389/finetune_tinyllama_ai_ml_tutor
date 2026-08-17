from datasets import load_dataset

def load(dataset_paths):
    dataset = load_dataset(
        "json",
        data_files=dataset_paths
    )
    
    return dataset