import os
from datetime import datetime
import json
from colorama import init, Fore, Style
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
import pygments.util
from anthropic import Anthropic
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
import traceback
import ast
import sys
from contextlib import redirect_stdout, redirect_stderr
import threading
import _thread
import time

load_dotenv()
init()

USER_COLOR = Fore.WHITE
CLAUDE_COLOR = Fore.BLUE
TOOL_COLOR = Fore.YELLOW
RESULT_COLOR = Fore.GREEN

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

conversation_history = []

current_df = None
figure_counter = 0

system_prompt = """
You are Claude, an AI data analyst for Datalore, powered by Anthropic's Claude-3.5-Sonnet model, integrated with a data analysis system. Your capabilities include:

1. Reading and displaying contents of various data file formats (CSV, Excel, JSON)
2. Data preprocessing and cleaning
3. Exploratory Data Analysis (EDA)
4. Statistical analysis
5. Data visualization
6. Machine learning model building and evaluation
7. Executing custom Python code

When interacting with the user:
- Help them analyze their data efficiently
- Offer suggestions for data exploration and insights
- Use the integrated tools to perform data analysis tasks as needed
- Provide clear and concise information about the analysis results
- Interpret and correctly handle user requests related to data analysis

Always strive to provide the most accurate, helpful, and detailed responses possible. If you're unsure about something, admit it and ask for clarification.

Answer the user's request using relevant tools (if they are available). Before calling a tool, analyze which tool is most appropriate and ensure you have all required parameters.
"""

def print_colored(text, color):
    print(f"{color}{text}{Style.RESET_ALL}")

def print_code(code, language):
    try:
        lexer = get_lexer_by_name(language, stripall=True)
        formatted_code = highlight(code, lexer, TerminalFormatter())
        print(formatted_code)
    except pygments.util.ClassNotFound:
        print_colored(f"Code (language: {language}):\n{code}", CLAUDE_COLOR)

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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_counter += 1
        filename = f"plot_{timestamp}_{figure_counter}.png"
        
        plt.savefig(filename)
        plt.close()
        
        return f"Visualization saved as {filename}"
    except Exception as e:
        return f"Error during visualization: {str(e)}"

def execute_code(code, timeout=30, max_output_length=10000):
    global current_df, figure_counter

    def analyze_code_safety(code):
        """Analyze the code for potentially unsafe operations."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if any(name.name == 'os' for name in node.names):
                        return False, "Importing 'os' module is not allowed for security reasons."
                if isinstance(node, (ast.Call, ast.Attribute)):
                    func_name = ''
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node, ast.Attribute):
                        func_name = node.attr
                    if func_name in ['eval', 'exec', 'compile']:
                        return False, f"Use of '{func_name}' is not allowed for security reasons."
            return True, "Code analysis passed."
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"

    def run_code_in_namespace(code, global_ns, local_ns):
        """Execute the code in a specific namespace and capture its output."""
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, global_ns, local_ns)
        
        return output_buffer.getvalue(), error_buffer.getvalue()

    def execute_with_timeout(code, global_ns, local_ns, timeout):
        """Execute the code with a timeout."""
        result = {"output": "", "error": "", "timed_out": False}
        
        def target():
            try:
                result["output"], result["error"] = run_code_in_namespace(code, global_ns, local_ns)
            except Exception as e:
                result["error"] = f"Error: {str(e)}\n{traceback.format_exc()}"

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            _thread.interrupt_main()
            thread.join()
            result["timed_out"] = True
            result["error"] = "Execution timed out"
        
        return result

    # step 1: Analyze code safety
    is_safe, safety_message = analyze_code_safety(code)
    if not is_safe:
        return {"output": "", "error": safety_message, "variables": {}}

    # step 2: Prepare the execution environment
    global_ns = {
        '__builtins__': __builtins__,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'datetime': datetime,
    }
    local_ns = {'current_df': current_df}

    # step 3: Replace 'df' with 'current_df' in the code
    modified_code = code.replace('df', 'current_df')

    # step 4: Execute the modified code with timeout
    result = execute_with_timeout(modified_code, global_ns, local_ns, timeout)

    # step 5: Process the results
    output = result["output"]
    error = result["error"]

    # truncating output if it's too long
    if len(output) > max_output_length:
        output = output[:max_output_length] + "\n... (output truncated)"
    
    # checking if the current_df has been modified
    if 'current_df' in local_ns and not local_ns['current_df'].equals(current_df):
        current_df = local_ns['current_df']  # Update the global current_df
        output += f"\n\nDataFrame modified. New shape: {current_df.shape}"
        output += f"\nFirst few rows:\n{current_df.head().to_string()}"
        
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"preprocessed_data_{timestamp}.csv"
        current_df.to_csv(csv_filename, index=False)
        output += f"\n\nPreprocessed data saved as: {csv_filename}"
    
    created_vars = {k: v for k, v in local_ns.items() if k not in global_ns and not k.startswith('_')}
    
    if plt.get_fignums():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_counter += 1
        filename = f"plot_{timestamp}_{figure_counter}.png"
        plt.savefig(filename)
        plt.close()
        output += f"\nPlot saved as {filename}"

    return {
        "output": output,
        "error": error,
        "variables": created_vars,
        "timed_out": result["timed_out"]
    }

tools = [
    {
        "name": "read_data",
        "description": "Read a data file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path of the file to read"
                },
                "file_type": {
                    "type": "string",
                    "description": "The type of the file (csv, excel, json)"
                }
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
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of preprocessing operations to perform"
                }
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
                "analysis_type": {
                    "type": "string",
                    "description": "Type of analysis to perform (e.g., 'summary', 'correlation', 'regression')"
                }
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
                "plot_type": {
                    "type": "string",
                    "description": "Type of plot to create (e.g., 'scatter', 'bar', 'histogram', 'line')"
                },
                "x_column": {
                    "type": "string",
                    "description": "Column to use for x-axis"
                },
                "y_column": {
                    "type": "string",
                    "description": "Column to use for y-axis (if applicable)"
                }
            },
            "required": ["plot_type", "x_column"]
        }
    },
    {
        "name": "execute_code",
        "description": "Execute custom Python code",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }
    }
]

def execute_tool(tool_name, tool_input):
    if tool_name == "read_data":
        return read_data(**tool_input)
    elif tool_name == "preprocess_data":
        return preprocess_data(**tool_input)
    elif tool_name == "analyze_data":
        return analyze_data(**tool_input)
    elif tool_name == "visualize_data":
        return visualize_data(**tool_input)
    elif tool_name == "execute_code":
        result = execute_code(**tool_input)
        return f"Output: {result['output']}\nError: {result['error']}\nVariables: {result['variables']}\nTimed out: {result['timed_out']}"
    else:
        return f"Unknown tool: {tool_name}"
    


def chat_with_claude(user_input):
    global conversation_history
    
    conversation_history.append({"role": "user", "content": user_input})
    
    messages = conversation_history.copy()
    
    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4000,
        system=system_prompt,
        messages=messages,
        tools=tools,
        tool_choice={"type": "auto"}
    )
    
    assistant_response = ""
    
    for content_block in response.content:
        if content_block.type == "text":
            assistant_response += content_block.text
            print_colored(f"\nClaude: {content_block.text}", CLAUDE_COLOR)
        elif content_block.type == "tool_use":
            tool_name = content_block.name
            tool_input = content_block.input
            tool_use_id = content_block.id
            
            print_colored(f"\nTool Used: {tool_name}", TOOL_COLOR)
            print_colored(f"Tool Input: {tool_input}", TOOL_COLOR)
            
            result = execute_tool(tool_name, tool_input)
            
            print_colored(f"Tool Result: {result}", RESULT_COLOR)
            
            conversation_history.append({"role": "assistant", "content": [content_block]})
            conversation_history.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result
                    }
                ]
            })
            
            tool_response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                system=system_prompt,
                messages=conversation_history,
                tools=tools,
                tool_choice={"type": "auto"}
            )
            
            for tool_content_block in tool_response.content:
                if tool_content_block.type == "text":
                    assistant_response += tool_content_block.text
                    print_colored(f"\nClaude: {tool_content_block.text}", CLAUDE_COLOR)
    
    conversation_history.append({"role": "assistant", "content": assistant_response})
    
    return assistant_response

def main():
    print_colored("Welcome to your AI-powered Data Analyst!\n", CLAUDE_COLOR)
    print_colored("I am Claude, and I can help you analyze data from various file formats.", CLAUDE_COLOR)
    print_colored("Just chat with me naturally about what you'd like to do and I'll do my best to assist you.", CLAUDE_COLOR)
    print_colored("You can ask me to read data files, preprocess data, perform statistical analysis, visualize data, and more.", CLAUDE_COLOR)
    print_colored("Type 'exit' to end the conversation.", CLAUDE_COLOR)
    
    while True:
        user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")
        if user_input.lower() == 'exit':
            print_colored("Thank you for using the AI Data Analyst. Goodbye!", CLAUDE_COLOR)
            break
        
        response = chat_with_claude(user_input)
        
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    print_colored(part, CLAUDE_COLOR)
                else:
                    lines = part.split('\n')
                    language = lines[0].strip() if lines else ""
                    code = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                    
                    if language and code:
                        print_code(code, language)
                    elif code:
                        print_colored(f"Code:\n{code}", CLAUDE_COLOR)
                    else:
                        print_colored(part, CLAUDE_COLOR)

if __name__ == "__main__":
    main()