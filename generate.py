from openai import OpenAI
import os
from utils import read_file_to_string, run_script
from config import files_to_generate  # Assuming files_to_generate is still relevant for total count
import datetime
import glob # To list files in a directory

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

system_instructions = read_file_to_string(f'prompts/system_instruction.txt')
if system_instructions is None:
    print('Cannot load system instructions.')
    raise 'error'


client = OpenAI()

success_counter = 0
types_dir = 'prompts/types' # Directory containing type prompts

type_prompt_files = glob.glob(os.path.join(types_dir, '*.txt')) # Get all .txt files in types_dir

if not type_prompt_files:
    print(f"No prompt files found in '{types_dir}'. Please add .txt files to this directory.")
    raise 'error'

for prompt_file_path in type_prompt_files:
    type_query = read_file_to_string(prompt_file_path) # Read prompt from each type file

    if type_query is None:
        print(f'Cannot load query from {prompt_file_path}')
        print('Skipping this type.') # Skip to the next type file instead of raising error, to continue processing other types.
        continue # Go to the next file in the loop

    
    current_query = type_query 
    for i in range (files_to_generate):
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system", "content": system_instructions
                },
                {
                    "role": "user", "content": current_query # Using the type-specific query
                }
            ]
        )

        generated_code = completion.choices[0].message.content
        print(f"Generated code for prompt file: {os.path.basename(prompt_file_path)}") # Indicate which prompt file is being processed
        print(generated_code) # Print the generated code

        now = datetime.datetime.now()
        filename = './seeds/' + now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + os.path.splitext(os.path.basename(prompt_file_path))[0] + '.py' # Include prompt file name in filename

        with open(filename, 'w') as f:
            f.write(generated_code)

        result = run_script(filename)
        #　Not runnable
        os.remove(filename)
        if result == 1:
            success_counter += 1

print(f'{success_counter} files generated successfully out of {len(type_prompt_files)} types.') # Report based on number of type prompt files.