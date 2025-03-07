import subprocess
import sys
import torch

def read_file_to_string(file_path):
    """
    Reads the entire content of a file into a single string.

    Args:
        file_path (str): The path to the file to be read.

    Returns:
        str: The entire content of the file as a string, or None if an error occurs
             (e.g., FileNotFoundError).
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            return content
    except FileNotFoundError:
        print(f"Error: File not found at path: {file_path}")
        return None  # Or you could raise the exception: raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        print(f"An unexpected error occurred while reading the file: {e}")
        return None # Or you could raise the exception: raise
    

def run_script(script_name):
    try:
        result = subprocess.run(["python", script_name], capture_output=True, text=True)
        print(f"Output from {script_name}:\n{result.stdout}")
        return 1
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script_name}: {e}")
        print(f"Stderr:\n{e.stderr}")
        return 0
    except FileNotFoundError:
        print(f"Error: Script '{script_name}' not found.")
        return -1
    except Exception as e:
         print(f"An unexpected error occurred: {e}")
         return -1
import torch

import torch

def compare_outputs(output_orig, output_trans):
    """
    Robustly compares two outputs, handling tensors, lists of tensors, and mixed structures.
    Relies on pytest for error reporting.
    """

    def _compare_elements(elem_orig, elem_trans, index_path=""):
        """
        Recursively compares elements, handling tensors, lists, tuples and dicts.
        Relies on pytest for error reporting.
        """
        print(f"Comparing at index_path: {index_path}")
        print(f"  Type elem_orig: {type(elem_orig)}")
        print(f"  Type elem_trans: {type(elem_trans)}")

        if isinstance(elem_orig, torch.Tensor) and isinstance(elem_trans, torch.Tensor):
            print(f"  Shape elem_orig: {elem_orig.shape}")
            print(f"  Shape elem_trans: {elem_trans.shape}")
            print(f"  Dtype elem_orig: {elem_orig.dtype}")
            print(f"  Dtype elem_trans: {elem_trans.dtype}")

            if elem_orig.numel() > 10:
                print(f"  Content elem_orig (first 10): {elem_orig.flatten()[:10]}")
                print(f"  Content elem_trans (first 10): {elem_trans.flatten()[:10]}")
            else:
                print(f"  Content elem_orig: {elem_orig}")
                print(f"  Content elem_trans: {elem_trans}")

            assert torch.allclose(elem_orig, elem_trans, rtol=1e-05, atol=1e-08)

        elif isinstance(elem_orig, list) and isinstance(elem_trans, list):
            assert len(elem_orig) == len(elem_trans)
            for i in range(len(elem_orig)):
                _compare_elements(elem_orig[i], elem_trans[i], index_path + f"[{i}]")

        elif isinstance(elem_orig, tuple) and isinstance(elem_trans, tuple):
            assert len(elem_orig) == len(elem_trans)
            for i in range(len(elem_orig)):
                _compare_elements(elem_orig[i], elem_trans[i], index_path + f"[{i}]")

        elif isinstance(elem_orig, dict) and isinstance(elem_trans, dict):
            print("  Comparing Dictionaries:")
            assert set(elem_orig.keys()) == set(elem_trans.keys()), f"Dict keys are different at {index_path}" # Check keys first
            for key in elem_orig:
                print(f"    Comparing key: '{key}'")
                print(f"      Type value_orig: {type(elem_orig[key])}") # Debug print for value types
                print(f"      Type value_trans: {type(elem_trans[key])}") # Debug print for value types
                _compare_elements(elem_orig[key], elem_trans[key], index_path + f"['{key}']") # Recursive call for dict values

        else:
            assert elem_orig == elem_trans

    print("Top-level comparison:")
    print(f"  Type output_orig: {type(output_orig)}")
    print(f"  Type output_trans: {type(output_trans)}")
    if isinstance(output_orig, list) and isinstance(output_trans, list):
        assert len(output_orig) == len(output_trans)
        for i in range(len(output_orig)):
            _compare_elements(output_orig[i], output_trans[i], f"[{i}]")
    else:
        _compare_elements(output_orig, output_trans)