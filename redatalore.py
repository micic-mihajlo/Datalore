import os
import asyncio
import json
import base64
import io
import re
import time
import datetime
import venv
import subprocess
import sys
import signal
from functools import partial
from PIL import Image
from dotenv import load_dotenv
from anthropic import Anthropic, APIStatusError, APIError
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

# Load environment variables
load_dotenv()

# Initialize Anthropic client
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
client = Anthropic(api_key=ANTHROPIC_API_KEY)

# Constants
MAINMODEL = "claude-3-5-sonnet-20240620"
MAX_CONTEXT_TOKENS = 1000000  # 1M tokens for context window
CONTINUATION_EXIT_PHRASE = "DATALORE_TASK_COMPLETE"
MAX_CONTINUATION_ITERATIONS = 25

# Initialize Rich console
console = Console()

# Global variables
conversation_history = []
current_df = None
figure_counter = 0
automode = False
running_processes = {}
token_usage = {
    'main_model': {'input': 0, 'output': 0},
    'tool_checker': {'input': 0, 'output': 0},
    'data_analyzer': {'input': 0, 'output': 0}
}

# System prompts
base_system_prompt = """
You are an AI data analyst assistant powered by Anthropic's Claude-3.5-Sonnet model, integrated with a sophisticated data analysis system. Your capabilities include:

1. Reading and processing various data file formats (CSV, Excel, JSON)
2. Data preprocessing and cleaning
3. Exploratory Data Analysis (EDA)
4. Statistical analysis and hypothesis testing
5. Advanced data visualization
6. Machine learning model building and evaluation
7. Executing custom Python code in an isolated environment

When interacting with users:
- Provide efficient and insightful data analysis
- Offer suggestions for data exploration and potential insights
- Use the integrated tools to perform complex data analysis tasks
- Provide clear, concise, and accurate information about analysis results
- Interpret and handle user requests related to data analysis professionally

Always strive to provide the most accurate, helpful, and detailed responses possible. If you're unsure about something, admit it and ask for clarification.

Use relevant tools when answering user requests. Before calling a tool, analyze which tool is most appropriate and ensure you have all required parameters.
"""

automode_system_prompt = """
You are currently in automode. Follow these guidelines:

1. Goal Setting:
   - Set clear, achievable data analysis goals based on the user's request.
   - Break down complex analytical tasks into smaller, manageable goals.

2. Goal Execution:
   - Work through goals systematically, using appropriate data analysis tools for each task.
   - Utilize data loading, preprocessing, analysis, and visualization functions as needed.
   - Always review data before and after any transformations.

3. Progress Tracking:
   - Provide regular updates on goal completion and overall analysis progress.
   - Use the iteration information to pace your work effectively.

4. Tool Usage:
   - Leverage all available data analysis tools to accomplish your goals efficiently.
   - Use execute_code for custom data operations when built-in tools are insufficient.

5. Error Handling:
   - If a data operation fails, analyze the error and attempt to resolve the issue.
   - For persistent errors, consider alternative approaches to achieve the analytical goal.

6. Automode Completion:
   - When all data analysis goals are completed, respond with "DATALORE_TASK_COMPLETE" to exit automode.
   - Provide a comprehensive summary of the analysis performed and insights gained.

7. Iteration Awareness:
   - You have access to this {iteration_info}.
   - Use this information to prioritize analytical tasks and manage time effectively.

Remember: Focus on completing the established data analysis goals efficiently and effectively. Avoid unnecessary conversations or requests for additional tasks outside the scope of the current analysis.
"""

def update_system_prompt(current_iteration=None, max_iterations=None):
    global base_system_prompt, automode_system_prompt
    chain_of_thought_prompt = """
    Before using a data analysis tool, think through these steps within <thinking></thinking> tags:
    1. Identify the most relevant tool for the current data analysis task.
    2. Consider each required parameter for the chosen tool.
    3. Determine if the user has provided or implied enough information for each parameter.
    4. If all parameters are available or can be reasonably inferred, proceed with the tool call.
    5. If any required parameter is missing, DO NOT use the tool. Instead, ask the user for the missing information.
    Do not ask for optional parameters if they're not provided.

    After using a tool, always interpret the results in the context of the overall data analysis goal.
    """
    if automode:
        iteration_info = f"You are on iteration {current_iteration} out of {max_iterations} in automode." if current_iteration and max_iterations else ""
        return base_system_prompt + "\n\n" + automode_system_prompt.format(iteration_info=iteration_info) + "\n\n" + chain_of_thought_prompt
    else:
        return base_system_prompt + "\n\n" + chain_of_thought_prompt

# Utility functions
def print_colored(text, color):
    console.print(f"[{color}]{text}[/{color}]")

def print_code(code, language):
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

def encode_image_to_base64(image_path):
    try:
        with Image.open(image_path) as img:
            max_size = (1024, 1024)
            img.thumbnail(max_size, Image.LANCZOS)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            return base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    except Exception as e:
        return f"Error encoding image: {str(e)}"

import subprocess
import pkg_resources

def setup_virtual_environment():
    venv_name = "data_analysis_env"
    venv_path = os.path.join(os.getcwd(), venv_name)
    newly_created = False
    if not os.path.exists(venv_path):
        venv.create(venv_path, with_pip=True)
        newly_created = True
    
    # Activate the virtual environment
    activate_script = os.path.join(venv_path, "Scripts", "activate.bat") if sys.platform == "win32" else os.path.join(venv_path, "bin", "activate")
    
    # Install required packages only if the environment was newly created
    if newly_created:
        install_required_packages(venv_path)
    
    return venv_path, activate_script

def install_required_packages(venv_path):
    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "scikit-learn": "sklearn",
        "openpyxl": "openpyxl",
    }
    pip_path = os.path.join(venv_path, "Scripts", "pip.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "pip")
    python_path = os.path.join(venv_path, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(venv_path, "bin", "python")

    for package, import_name in packages.items():
        try:
            # Check if the package is installed and importable
            subprocess.check_call([python_path, "-c", f"import {import_name}"])
            print(f"{package} is already installed.")
        except subprocess.CalledProcessError:
            print(f"Installing {package}...")
            subprocess.check_call([pip_path, "install", package])

async def execute_code(code, timeout=30):
    global running_processes
    venv_path, activate_script = setup_virtual_environment()
    
    process_id = f"process_{len(running_processes)}"
    
    # Prepare the Python script with necessary imports
    script_content = f"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# User's code
{code}
"""
    
    with open(f"{process_id}.py", "w") as f:
        f.write(script_content)
    
    if sys.platform == "win32":
        command = f'"{activate_script}" && python {process_id}.py'
    else:
        command = f'source "{activate_script}" && python {process_id}.py'
    
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        preexec_fn=None if sys.platform == "win32" else os.setsid
    )
    
    running_processes[process_id] = process
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode()
        stderr = stderr.decode()
        return_code = process.returncode
    except asyncio.TimeoutError:
        stdout = "Process started and running in the background."
        stderr = ""
        return_code = "Running"
    
    execution_result = f"Process ID: {process_id}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}\n\nReturn Code: {return_code}"
    return process_id, execution_result

# Data analysis functions
def read_data(file_path, file_type):
    global current_df
    try:
        if file_type == "csv":
            current_df = pd.read_csv(file_path)
        elif file_type == "excel":
            current_df = pd.read_excel(file_path)
        elif file_type == "json":
            current_df = pd.read_json(file_path)
        else:
            return "Unsupported file type"
        return f"Data read successfully. Shape: {current_df.shape}\n\nFirst few rows:\n{current_df.head().to_string()}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def preprocess_data(operations):
    global current_df
    if current_df is None:
        return "No data loaded. Please read a data file first."
    try:
        for operation in operations:
            if operation == "drop_na":
                current_df = current_df.dropna()
            elif operation == "fill_na_mean":
                current_df = current_df.fillna(current_df.mean())
            elif operation == "normalize":
                current_df = (current_df - current_df.mean()) / current_df.std()
        return f"Preprocessing completed. New shape: {current_df.shape}"
    except Exception as e:
        return f"Error during preprocessing: {str(e)}"

def analyze_data(analysis_type):
    global current_df
    if current_df is None:
        return "No data loaded. Please read a data file first."
    try:
        if analysis_type == "summary":
            return current_df.describe().to_string()
        elif analysis_type == "correlation":
            return current_df.corr().to_string()
        elif analysis_type == "regression":
            X = current_df.iloc[:, :-1]
            y = current_df.iloc[:, -1]
            model = LinearRegression().fit(X, y)
            return f"Regression coefficients: {model.coef_}"
        else:
            return f"Unsupported analysis type: {analysis_type}"
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def visualize_data(plot_type, x_column, y_column=None):
    global current_df, figure_counter
    if current_df is None:
        return "No data loaded. Please read a data file first."
    try:
        plt.figure(figsize=(10, 6))
        if plot_type == "scatter":
            sns.scatterplot(data=current_df, x=x_column, y=y_column)
        elif plot_type == "bar":
            sns.barplot(data=current_df, x=x_column, y=y_column)
        elif plot_type == "histogram":
            sns.histplot(data=current_df, x=x_column)
        elif plot_type == "line":
            sns.lineplot(data=current_df, x=x_column, y=y_column)
        plt.title(f"{plot_type.capitalize()} plot")
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_counter += 1
        filename = f"plot_{timestamp}_{figure_counter}.png"
        
        plt.savefig(filename)
        plt.close()
        
        return f"Visualization saved as {filename}"
    except Exception as e:
        return f"Error during visualization: {str(e)}"

async def execute_code(code, timeout=30):
    global running_processes
    venv_path, activate_script = setup_virtual_environment()
    
    process_id = f"process_{len(running_processes)}"
    
    with open(f"{process_id}.py", "w") as f:
        f.write(code)
    
    if sys.platform == "win32":
        command = f'"{activate_script}" && python {process_id}.py'
    else:
        command = f'source "{activate_script}" && python {process_id}.py'
    
    process = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        shell=True,
        preexec_fn=None if sys.platform == "win32" else os.setsid
    )
    
    running_processes[process_id] = process
    
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        stdout = stdout.decode()
        stderr = stderr.decode()
        return_code = process.returncode
    except asyncio.TimeoutError:
        stdout = "Process started and running in the background."
        stderr = ""
        return_code = "Running"
    
    execution_result = f"Process ID: {process_id}\n\nStdout:\n{stdout}\n\nStderr:\n{stderr}\n\nReturn Code: {return_code}"
    return process_id, execution_result

def stop_process(process_id):
    global running_processes
    if process_id in running_processes:
        process = running_processes[process_id]
        if sys.platform == "win32":
            process.terminate()
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        del running_processes[process_id]
        return f"Process {process_id} has been stopped."
    else:
        return f"No running process found with ID {process_id}."

# Tool definitions
tools = [
    {
        "name": "read_data",
        "description": "Read a data file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "The path of the file to read"},
                "file_type": {"type": "string", "description": "The type of the file (csv, excel, json)"}
            },
            "required": ["file_path", "file_type"]
        }
    },
    {
        "name": "preprocess_data",
        "description": "Preprocess and clean the data",
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {"type": "array", "items": {"type": "string"}, "description": "List of preprocessing operations to perform"}
            },
            "required": ["operations"]
        }
    },
    {
        "name": "analyze_data",
        "description": "Perform statistical analysis on the data",
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis_type": {"type": "string", "description": "Type of analysis to perform (e.g., 'summary', 'correlation', 'regression')"}
            },
            "required": ["analysis_type"]
        }
    },
    {
        "name": "visualize_data",
        "description": "Create data visualizations",
        "input_schema": {
            "type": "object",
            "properties": {
                "plot_type": {"type": "string", "description": "Type of plot to create (e.g., 'scatter', 'bar', 'histogram', 'line')"},
                "x_column": {"type": "string", "description": "Column to use for x-axis"},
                "y_column": {"type": "string", "description": "Column to use for y-axis (if applicable)"}
            },
            "required": ["plot_type", "x_column"]
        }
    },
    {
        "name": "execute_code",
        "description": "Execute custom Python code in the data analysis environment",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "required": ["code"]
        }
    },
    {
        "name": "stop_process",
        "description": "Stop a running process by its ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "process_id": {"type": "string", "description": "The ID of the process to stop"}
            },
            "required": ["process_id"]
        }
    }
]

# Tool execution function
async def execute_tool(tool_name, tool_input):
    try:
        if tool_name == "read_data":
            return read_data(**tool_input)
        elif tool_name == "preprocess_data":
            return preprocess_data(**tool_input)
        elif tool_name == "analyze_data":
            return analyze_data(**tool_input)
        elif tool_name == "visualize_data":
            return visualize_data(**tool_input)
        elif tool_name == "execute_code":
            process_id, execution_result = await execute_code(**tool_input)
            analysis = await analyze_code_execution(tool_input["code"], execution_result)
            return f"{execution_result}\n\nAnalysis:\n{analysis}"
        elif tool_name == "stop_process":
            return stop_process(**tool_input)
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error executing tool {tool_name}: {str(e)}"

async def analyze_code_execution(code, execution_result):
    try:
        system_prompt = f"""
        You are an AI data analysis code execution analyst. Your task is to analyze the provided code and its execution result from the data analysis environment, then provide a concise summary of what worked, what didn't work, and any important observations. Follow these steps:

        1. Review the code that was executed in the data analysis environment:
        {code}

        2. Analyze the execution result:
        {execution_result}

        3. Provide a brief summary of:
           - What parts of the code executed successfully
           - Any errors or unexpected behavior encountered
           - Potential improvements or fixes for issues
           - Any important observations about the code's performance or output
           - If the execution timed out, explain what this might mean (e.g., long-running process, infinite loop)

        Be concise and focus on the most important aspects of the code execution within the data analysis context.

        IMPORTANT: PROVIDE ONLY YOUR ANALYSIS AND OBSERVATIONS. DO NOT INCLUDE ANY PREFACING STATEMENTS OR EXPLANATIONS OF YOUR ROLE.
        """

        response = client.messages.create(
            model=MAINMODEL,
            max_tokens=1000,
            system=system_prompt,
            messages=[
                {"role": "user", "content": f"Analyze this code execution:\n\nCode:\n{code}\n\nExecution Result:\n{execution_result}"}
            ]
        )

        # Update token usage for data analyzer
        token_usage['data_analyzer']['input'] += response.usage.input_tokens
        token_usage['data_analyzer']['output'] += response.usage.output_tokens

        return response.content[0].text

    except Exception as e:
        console.print(f"Error in AI code execution analysis: {str(e)}", style="bold red")
        return f"Error analyzing code execution: {str(e)}"

async def chat_with_claude(user_input, image_path=None, current_iteration=None, max_iterations=None):
    global conversation_history, automode, token_usage

    current_conversation = []

    if image_path:
        console.print(Panel(f"Processing image at path: {image_path}", title="Image Processing", style="yellow"))
        image_base64 = encode_image_to_base64(image_path)

        if image_base64.startswith("Error"):
            console.print(Panel(f"Error encoding image: {image_base64}", title="Error", style="bold red"))
            return "I'm sorry, there was an error processing the image. Please try again.", False

        image_message = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": image_base64
                    }
                },
                {
                    "type": "text",
                    "text": f"User input for image: {user_input}"
                }
            ]
        }
        current_conversation.append(image_message)
        console.print(Panel("Image added to conversation", title="Image Added", style="green"))
    else:
        current_conversation.append({"role": "user", "content": user_input})

    messages = conversation_history + current_conversation

    try:
        response = client.messages.create(
            model=MAINMODEL,
            max_tokens=4096,
            system=update_system_prompt(current_iteration, max_iterations),
            messages=messages,
            tools=tools,
            tool_choice={"type": "auto"}
        )
        # Update token usage
        token_usage['main_model']['input'] += response.usage.input_tokens
        token_usage['main_model']['output'] += response.usage.output_tokens
        
        display_token_usage()
    except APIStatusError as e:
        if e.status_code == 429:
            console.print(Panel("Rate limit exceeded. Retrying after a short delay...", title="API Error", style="bold yellow"))
            await asyncio.sleep(5)
            return await chat_with_claude(user_input, image_path, current_iteration, max_iterations)
        else:
            console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
            return "I'm sorry, there was an error communicating with the AI. Please try again.", False
    except APIError as e:
        console.print(Panel(f"API Error: {str(e)}", title="API Error", style="bold red"))
        return "I'm sorry, there was an error communicating with the AI. Please try again.", False

    assistant_response = ""
    exit_continuation = False
    tool_uses = []

    for content_block in response.content:
        if content_block.type == "text":
            assistant_response += content_block.text
            if CONTINUATION_EXIT_PHRASE in content_block.text:
                exit_continuation = True
        elif content_block.type == "tool_use":
            tool_uses.append(content_block)

    console.print(Panel(Markdown(assistant_response), title="Claude's Response", border_style="blue"))

    for tool_use in tool_uses:
        tool_name = tool_use.name
        tool_input = tool_use.input
        tool_use_id = tool_use.id

        console.print(Panel(f"Tool Used: {tool_name}", style="green"))
        console.print(Panel(f"Tool Input: {json.dumps(tool_input, indent=2)}", style="green"))

        try:
            result = await execute_tool(tool_name, tool_input)
            console.print(Panel(result, title="Tool Result", style="green"))
        except Exception as e:
            result = f"Error executing tool: {str(e)}"
            console.print(Panel(result, title="Tool Execution Error", style="bold red"))

        current_conversation.append({
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_use_id,
                    "name": tool_name,
                    "input": tool_input
                }
            ]
        })

        current_conversation.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result
                }
            ]
        })

        messages = conversation_history + current_conversation

        try:
            tool_response = client.messages.create(
                model=MAINMODEL,
                max_tokens=4000,
                system=update_system_prompt(current_iteration, max_iterations),
                messages=messages,
                tools=tools,
                tool_choice={"type": "auto"}
            )
            # Update token usage for tool checker
            token_usage['tool_checker']['input'] += tool_response.usage.input_tokens
            token_usage['tool_checker']['output'] += tool_response.usage.output_tokens

            display_token_usage()

            tool_checker_response = ""
            for tool_content_block in tool_response.content:
                if tool_content_block.type == "text":
                    tool_checker_response += tool_content_block.text
            console.print(Panel(Markdown(tool_checker_response), title="Claude's Response to Tool Result", border_style="blue"))
            assistant_response += "\n\n" + tool_checker_response
        except APIError as e:
            error_message = f"Error in tool response: {str(e)}"
            console.print(Panel(error_message, title="Error", style="bold red"))
            assistant_response += f"\n\n{error_message}"

    if assistant_response:
        current_conversation.append({"role": "assistant", "content": assistant_response})

    conversation_history = messages + [{"role": "assistant", "content": assistant_response}]

    return assistant_response, exit_continuation

def reset_conversation():
    global conversation_history, token_usage
    conversation_history = []
    token_usage = {model: {'input': 0, 'output': 0} for model in token_usage}
    console.print(Panel("Conversation history and token counts have been reset.", title="Reset", style="bold green"))
    display_token_usage()

def display_token_usage():
    console.print("\nToken Usage:")
    total_input = 0
    total_output = 0
    
    for model, tokens in token_usage.items():
        total = tokens['input'] + tokens['output']
        percentage = (total / MAX_CONTEXT_TOKENS) * 100
        
        total_input += tokens['input']
        total_output += tokens['output']

        console.print(f"{model.capitalize()}:")
        console.print(f"  Input: {tokens['input']}, Output: {tokens['output']}, Total: {total}")
        console.print(f"  Percentage of context window used: {percentage:.2f}%")
        
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(bar_width=50),
                      TextColumn("[progress.percentage]{task.percentage:>3.0f}%")) as progress:
            progress.add_task(f"Context window usage", total=100, completed=percentage)

    grand_total = total_input + total_output
    total_percentage = (grand_total / MAX_CONTEXT_TOKENS) * 100

    input_cost = (total_input / 1_000_000) * 3.00
    output_cost = (total_output / 1_000_000) * 15.00
    total_cost = input_cost + output_cost

    console.print(f"\nTotal Token Usage: Input: {total_input}, Output: {total_output}, Grand Total: {grand_total}, Cost: ${total_cost:.3f}")
    console.print("\n")

def save_chat():
    now = datetime.datetime.now()
    filename = f"DataLore_Chat_{now.strftime('%Y%m%d_%H%M')}.md"
    
    formatted_chat = "# DataLore AI Data Analyst Chat Log\n\n"
    for message in conversation_history:
        if message['role'] == 'user':
            formatted_chat += f"## User\n\n{message['content']}\n\n"
        elif message['role'] == 'assistant':
            if isinstance(message['content'], str):
                formatted_chat += f"## DataLore\n\n{message['content']}\n\n"
            elif isinstance(message['content'], list):
                for content in message['content']:
                    if content['type'] == 'tool_use':
                        formatted_chat += f"### Tool Use: {content['name']}\n\n```json\n{json.dumps(content['input'], indent=2)}\n```\n\n"
                    elif content['type'] == 'text':
                        formatted_chat += f"## DataLore\n\n{content['text']}\n\n"
        elif message['role'] == 'user' and isinstance(message['content'], list):
            for content in message['content']:
                if content['type'] == 'tool_result':
                    formatted_chat += f"### Tool Result\n\n```\n{content['content']}\n```\n\n"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(formatted_chat)
    
    return filename

async def main():
    global automode
    console.print(Panel("Welcome to DataLore: Your AI-powered Data Analyst!", title="Welcome", style="bold green"))
    console.print("Type 'exit' to end the conversation.")
    console.print("Type 'image' to include an image in your message.")
    console.print("Type 'automode [number]' to enter Autonomous mode with a specific number of iterations.")
    console.print("Type 'reset' to clear the conversation history.")
    console.print("Type 'save chat' to save the conversation to a Markdown file.")
    console.print("While in automode, press Ctrl+C at any time to exit and return to regular chat.")

    while True:
        user_input = console.input("[bold cyan]You:[/bold cyan] ")

        if user_input.lower() == 'exit':
            console.print(Panel("Thank you for using DataLore. Goodbye!", title="Goodbye", style="bold green"))
            break

        if user_input.lower() == 'reset':
            reset_conversation()
            continue

        if user_input.lower() == 'save chat':
            filename = save_chat()
            console.print(Panel(f"Chat saved to {filename}", title="Chat Saved", style="bold green"))
            continue

        if user_input.lower() == 'image':
            image_path = console.input("[bold cyan]Drag and drop your image here, then press enter:[/bold cyan] ").strip().replace("'", "")

            if os.path.isfile(image_path):
                user_input = console.input("[bold cyan]You (prompt for image):[/bold cyan] ")
                response, _ = await chat_with_claude(user_input, image_path)
            else:
                console.print(Panel("Invalid image path. Please try again.", title="Error", style="bold red"))
                continue
        elif user_input.lower().startswith('automode'):
            try:
                parts = user_input.split()
                max_iterations = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else MAX_CONTINUATION_ITERATIONS

                automode = True
                console.print(Panel(f"Entering automode with {max_iterations} iterations. Please provide the goal of the automode.", title="Automode", style="bold yellow"))
                console.print(Panel("Press Ctrl+C at any time to exit the automode loop.", style="bold yellow"))
                user_input = console.input("[bold cyan]You:[/bold cyan] ")

                iteration_count = 0
                try:
                    while automode and iteration_count < max_iterations:
                        response, exit_continuation = await chat_with_claude(user_input, current_iteration=iteration_count+1, max_iterations=max_iterations)

                        if exit_continuation or CONTINUATION_EXIT_PHRASE in response:
                            console.print(Panel("Automode completed.", title="Automode", style="green"))
                            automode = False
                        else:
                            console.print(Panel(f"Continuation iteration {iteration_count + 1} completed. Press Ctrl+C to exit automode.", title="Automode", style="yellow"))
                            user_input = "Continue with the next step in the data analysis. Or STOP by saying 'DATALORE_TASK_COMPLETE' if you think you've achieved the results established in the original request."
                        iteration_count += 1

                        if iteration_count >= max_iterations:
                            console.print(Panel("Max iterations reached. Exiting automode.", title="Automode", style="bold red"))
                            automode = False
                except KeyboardInterrupt:
                    console.print(Panel("\nAutomode interrupted by user. Exiting automode.", title="Automode", style="bold red"))
                    automode = False
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further with your data analysis?"})
            except KeyboardInterrupt:
                console.print(Panel("\nAutomode interrupted by user. Exiting automode.", title="Automode", style="bold red"))
                automode = False
                if conversation_history and conversation_history[-1]["role"] == "user":
                    conversation_history.append({"role": "assistant", "content": "Automode interrupted. How can I assist you further with your data analysis?"})

            console.print(Panel("Exited automode. Returning to regular chat.", style="green"))
        else:
            response, _ = await chat_with_claude(user_input)

if __name__ == "__main__":
    asyncio.run(main())