# code_executor.py
import ast
import importlib.util
import sys
from ast_transformer import TorchMathTransformer  # Assuming this is available in your environment
import torch
import torch.nn as nn
import os
import inspect

GENERATED_DIR = "generated"  # Directory to save transformed code
SAVE_TRANSFORMED_CODE = True # Flag to control saving transformed code
USE_TORCH_COMPILE = True  # Flag to control using torch.compile


def transform_code(code):
    """Transforms Python code using TorchMathTransformer."""
    tree = ast.parse(code)
    transformer = TorchMathTransformer()
    transformed_tree = transformer.visit(tree)
    ast.fix_missing_locations(transformed_tree)
    transformed_code = ast.unparse(transformed_tree)
    return transformed_code

def execute_code(code, global_namespace=None, torch_compiled=False):
    """Executes Python code, either uncompiled or torch.compiled, and returns outputs and input tensors.
       Optionally compiles the model with torch.compile if torch_compiled=True.
    """
    if global_namespace is None:
        global_namespace = {}
    try:
        exec(code, global_namespace)

        # Assuming the code defines 'PtModule' and 'input_tensors'
        math_module = global_namespace.get('PtModule')
        input_tensors = global_namespace.get('input_tensors')

        if math_module and input_tensors and isinstance(input_tensors, list) and input_tensors:
            model = math_module()
            num_args = len(inspect.getfullargspec(model.forward).args)
            print(num_args)
            if torch_compiled and USE_TORCH_COMPILE: # Only compile if torch_compiled is True and global flag is True
                try:
                    model = torch.compile(model, backend="inductor")
                    print("Model compiled with torch.compile!")
                except Exception as compile_e:
                    print(f"Error during torch.compile (continuing without compilation): {compile_e}")
                    # Continue without compilation if torch.compile fails, for robustness


            output= []


           
            for i in range(len(input_tensors)):

                
                if num_args>2:
                    output.append(model.forward(*input_tensors[i]))
                else: 
                    output.append(model(input_tensors[i]))
                return output,  input_tensors
        else:
            return  None, input_tensors # Indicate failure to extract output but return inputs if available

    except Exception as e:
        print(f"Error executing code (uncompiled/torch_compiled={torch_compiled}): {e}")
        return  None, None

def run_code_and_compare(filename):
    """
    Reads code from a file, transforms it, and executes both original and transformed code
    in uncompiled and torch.compiled versions. Compares the outputs and returns results.
    Saves transformed code to the 'generated' directory if SAVE_TRANSFORMED_CODE is True.
    """
    with open(filename, "r") as file:
        original_code = file.read()

    transformed_code = transform_code(original_code)

    # --- Save Transformed Code ---
    if SAVE_TRANSFORMED_CODE:
        os.makedirs(GENERATED_DIR, exist_ok=True) # Create generated directory if it doesn't exist
        base_filename = os.path.basename(filename)
        name_without_extension = os.path.splitext(base_filename)[0]
        transformed_filename = os.path.join(GENERATED_DIR, f"{name_without_extension}_transformed.py")
        try:
            with open(transformed_filename, "w") as f:
                f.write(transformed_code)
            print(f"Transformed source code saved to {transformed_filename}")
        except Exception as e:
            print(f"Error saving transformed source code to {transformed_filename}: {e}")


    original_output_uncompiled, _ = execute_code(original_code)
    transformed_output_uncompiled,  _ = execute_code(transformed_code, global_namespace={'PtModule': nn.Module}) # Changed Module name to PtModule

    original_output_torch_compiled,  _ = execute_code(original_code, torch_compiled=True)
    transformed_output_torch_compiled,  _ = execute_code(transformed_code, global_namespace={'PtModule': nn.Module}, torch_compiled=True) # Changed Module name to PtModule


    return original_code, transformed_code, \
           original_output_uncompiled, transformed_output_uncompiled, \
           original_output_torch_compiled, transformed_output_torch_compiled
