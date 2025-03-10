# test_script.py
import pytest
from code_executor import run_code_and_compare, USE_TORCH_COMPILE
import torch
import os
from  utils import compare_outputs 
TEST_FILES_DIR = "./seeds"  # Directory containing test files

def assert_outputs_close(output_orig, output_trans, execution_type, test_file_path):
    """Helper function to assert if outputs are close, with detailed error messages."""
    #assert output_orig is not None, f"Original {execution_type} code execution failed for {test_file_path}."
    #assert output_trans is not None, f"Transformed {execution_type} code execution failed for {test_file_path}."
    
    compare_outputs(output_orig, output_trans)



def get_test_files(directory):
    """Returns a list of test file paths from the given directory."""
    test_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            test_files.append(os.path.join(directory, filename))
    return test_files

@pytest.mark.parametrize("test_file_path", get_test_files(TEST_FILES_DIR))
def test_code_transformation_directory(test_file_path):
    """Tests code transformation for all Python files in the test directory, including torch.compile."""
    (original_code, transformed_code,
     original_output_uncompiled, transformed_output_uncompiled,
     original_output_torch_compiled, transformed_output_torch_compiled) = run_code_and_compare(test_file_path)

    assert_outputs_close(original_output_uncompiled, transformed_output_uncompiled, "Uncompiled", test_file_path)
    if USE_TORCH_COMPILE: # Only test torch.compile if it's enabled globally
        assert_outputs_close(original_output_uncompiled,original_output_torch_compiled, "Original", test_file_path)
        assert_outputs_close(original_output_torch_compiled, transformed_output_torch_compiled, "Torch Compiled", test_file_path)
        
