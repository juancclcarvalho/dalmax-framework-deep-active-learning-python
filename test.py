import os
import glob

def list_folders_with_pattern(initial_path, pattern="*"):
    """Lists folders within a given path that match a specific pattern.

    Args:
        initial_path: The initial directory path.
        pattern: The pattern to match (default is '*', matching all).

    Returns:
        A list of folder paths that match the pattern.
    """

    matching_files = glob.glob(os.path.join(initial_path, pattern))
    # Filter only folders
    folders = [file for file in matching_files if os.path.isdir(file)]
    return folders

# Example usage:
path = "results/dalmax/daninhas_balanceado/"
pattern = "SEED*"  # List only folders starting with "SEED"
print(list_folders_with_pattern(path, pattern))