# Node Integration Skills: Adding New Tools to MoDrAg

This document provides a standardized procedure for integrating new tool functions (nodes) into the MoDrAg chatbot. Follow these steps to ensure consistency in logging, dependency management, and tool registration.

## Phase 1: Preparation & Dependency Audit
Before integrating the code, identify all external libraries used by the new function.
- [ ] **Audit Imports**: List all `import` statements required for the function.
- [ ] **Environment Check**: Verify that all dependencies are installed in the local environment.
- [ ] **Sourcing**: If the function is currently in a standalone file (e.g., `similarity.py`), prepare to migrate the logic into the main function modules.

## Phase 2: Code Refactoring & Migration
The chatbot uses a `print_flag` system to control verbosity. All functions must adhere to this pattern.

### 1. Module Selection
Determine which category the tool belongs to and move the function into the corresponding file:
- **Molecules**: `modrag_molecule_functions.py`
- **Properties**: `modrag_property_functions.py`
- **Proteins**: `modrag_protein_functions.py`

### 2. Implementation of `print_flag`
Ensure the function handles console output as follows:
- **Node Declaration**: The initial `print("tool name")` and separator line (e.g., `print('===================================================')`) **must remain outside** any flags. These are used to identify which tool is running.
- **Detailed Logging**: All other `print()` statements (debug info, intermediate results, counters, etc.) **must be wrapped** in an `if print_flag:` block.
  - *Example*:
    ```python
    print("my_tool_node")  # KEEP
    print('===================================================') # KEEP
    
    # ... logic ...
    
    if print_flag:
        print(f"Processed {count} items") # WRAP
    ```

## Phase 3: Chatbot Registration (`modrag.py`)
The function must be registered in `modrag.py` to be accessible to the LLM.

### 1. Import
- [ ] Add the new function to the import list from the chosen module.
  - *Example*: `from modrag_molecule_functions import ..., new_function_node`

### 2. Global Tools List
- [ ] Add the function object to the `tools` list. This list is used to generate the system prompt.

### 3. Execution Mapping
- [ ] Add the function to the `available_functions` dictionary.
  - *Example*: `'new_function_node': new_function_node`

### 4. API Configuration
- [ ] Add the function object to the `tools` list within the `client.chat()` call to enable tool-calling capabilities.

## Phase 4: Verification
- [ ] **Verbose Test**: Run with `python modrag.py --print`. Verify that the tool header appears and all flagged logs are visible.
- [ ] **Silent Test**: Run without `--print`. Verify that ONLY the tool header is printed.
- [ ] **Functional Test**: Trigger the tool via the chatbot to verify correctness of logic, output strings, and any generated images.
