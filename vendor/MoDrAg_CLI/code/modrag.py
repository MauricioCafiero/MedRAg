# Import torch BEFORE any heavy native libs (rdkit via the node modules, meeko/
# autodock-vina via vina_dock, ...). If Metal/MPS initializes after those libs are
# loaded, the Metal backend is corrupted and the GPT finetune segfaults (SIGSEGV,
# exit 139) on the first MPS tensor allocation in train_gpt (CafChemGPT.py:715).
# Importing torch first lets Metal init cleanly and coexist with the rest. This is
# the only place that needs the change: every other entry point (standalone
# finetune_gpt, the original CafChem repo) already imports torch before the heavy
# libs, which is why this only ever crashed inside modrag. See memory note
# gpt-finetune-chembl220-crash.
import torch  # noqa: E402  -- MUST stay first; do not move below the node/vina imports
import os
from PIL import Image
from collections import Counter
from typing import Annotated, TypedDict
import time, sys
from ollama import Client as ollama_client
from rich.console import Console
from rich.markdown import Markdown
from markdown_render import to_unicode_markdown

from modrag_protein_functions import uniprot_node, listbioactives_node, getbioactives_node, find_PDBID_node, target_node, docking_node, pdb_node, predict_node, check_nearby_molecules, get_pdb_file
from modrag_molecule_functions import name_node, smiles_node, related_node, structure_node, canonical_node
from modrag_property_functions import substitution_node, pharmfeature_node, lipinski_node, get_qed, similarity_node
from modrag_shell_functions import shell_node
from admet_node import admet_node
from vina_dock import blind_dock_agent
from gpt_node import gpt_node
import modrag_memory

console = Console(width=80)
import modrag_protein_functions
import modrag_molecule_functions
import modrag_property_functions
import modrag_shell_functions
import admet_node as admet_node_module
import subs_code
import vina_dock
import gpt_node as gpt_node_module
import finetune_gpt
import CafChemGPT

# get the print flag from the command line arguments, if they exist, otherwise set it to False
if len(sys.argv) > 1 and sys.argv[1] == '--print':
    print_flag = True
else:
    print_flag = False

# Set print_flag in all imported modules
modrag_protein_functions.print_flag = print_flag
modrag_molecule_functions.print_flag = print_flag
modrag_property_functions.print_flag = print_flag
modrag_shell_functions.print_flag = print_flag
admet_node_module.print_flag = print_flag
subs_code.print_flag = print_flag
modrag_memory.print_flag = print_flag
gpt_node_module.print_flag = print_flag
finetune_gpt.print_flag = print_flag
CafChemGPT.print_flag = print_flag

# When the current session's work began (time.time()). Used by the `memory`
# keyword to scope which PDB/SDF/CSV files were touched *this* session.
# Reset on every save and on start_chat so each memory covers a clean window.
session_start_time = time.time()

# Proximity fallback for blind docking: when True, blind_dock docks the top-2
# detected pockets and, if the best-score pocket's pose is NOT near a
# co-crystallized ligand, switches to the other pocket when it IS near one
# (proximity preferred over Vina score).
vina_dock.FALLBACK_TO_SECOND_POCKET = True

tools = [uniprot_node, listbioactives_node, getbioactives_node, find_PDBID_node, target_node, docking_node, pdb_node,
         name_node, smiles_node, related_node, structure_node, canonical_node,
         substitution_node, pharmfeature_node, lipinski_node, get_qed, predict_node, similarity_node,
         check_nearby_molecules, get_pdb_file, blind_dock_agent, gpt_node,
         shell_node, admet_node]

#get ket from shell variable $OLLAMA_API_KEY
ollama_key = os.getenv('OLLAMA_API_KEY')

# models = ['deepseek-v3.1:671b', 'gpt-oss:120b', 'gpt-oss:20b',
#           'devstral-2:123b', 'cogito-2.1:671b',
#           'nemotron-3-nano:30b', 'gemini-3-flash-preview',
#           'kimi-k2:1t', 'kimi-k2.5', 'gemma4:31b-cloud']

models = [
    'gemma4:31b', 'glm-5.2', 'kimi-k2.7-code',
    'deepseek-v4-pro', 'qwen3.5:397b',
]

model = models[1] # default model to use for chat

sys_message = f'''
You are a drug discovery assistant names Modrag. You have access to the 
following tools: {', '.join([tool.__name__ for tool in tools])}. Answer the 
user questions using these tools and your knowledge of drug discovery, 
chemistry, and biology. Add some enriching information to the tool results from your own 
knowledge, but do not make up any information and be concise. If you do not 
know the answer, say "I don't know" and do not make up an answer.
'''

global messages
messages = [{'role': 'system', 'content': sys_message}]

def start_chat():
  '''
  Initializes a new chat session by resetting the chat history, reasoning, and messages.
  '''
  global chat_history, messages, reasoning, session_start_time
  chat_history = []
  reasoning = []
  messages = [{'role': 'system', 'content': sys_message}]
  session_start_time = time.time()

def chat_turn(prompt: str):
  '''
  Handles a single turn of the chat by sending the user's prompt to the Ollama API,
  processing the response, and executing any tool calls if present.
  '''
  global chat_history, messages, reasoning
  
  client = ollama_client(host = 'https://ollama.com',
            headers={'Authorization': f'Bearer {ollama_key}'})

  available_functions = {
    'uniprot_node': uniprot_node,
    'listbioactives_node': listbioactives_node,
    'getbioactives_node': getbioactives_node,
    'find_PDBID_node': find_PDBID_node,
    'target_node': target_node,
    'docking_node': docking_node,
    'pdb_node': pdb_node,
    'name_node': name_node,
    'smiles_node': smiles_node,
    'related_node': related_node,
    'structure_node': structure_node, 
    'canonical_node': canonical_node,
    'substitution_node': substitution_node,
    'pharmfeature_node': pharmfeature_node,
    'lipinski_node': lipinski_node,
    'get_qed': get_qed,
    'predict_node': predict_node,
    'similarity_node': similarity_node,
    'check_nearby_molecules': check_nearby_molecules,
    'get_pdb_file': get_pdb_file,
    'blind_dock_agent': blind_dock_agent,
    'gpt_node': gpt_node,
    'shell_node': shell_node,
    'admet_node': admet_node
  }

  messages.append({'role': 'user', 'content': prompt})

  while True:
      response = client.chat(
          model=model,
          messages=messages,
          tools=[uniprot_node, listbioactives_node, getbioactives_node, find_PDBID_node, target_node, docking_node, pdb_node,
         name_node, smiles_node, related_node, structure_node, canonical_node,
         substitution_node, pharmfeature_node, lipinski_node, get_qed, predict_node, similarity_node,
         check_nearby_molecules, get_pdb_file, blind_dock_agent, gpt_node,
         shell_node, admet_node],
          think=True,
      )
      messages.append(response.message)
      if print_flag:
          print('------------------------------------------------------------------------')
          print("Thinking: ", response.message.thinking)
          print('------------------------------------------------------------------------')
          print("Content: ", response.message.content)
          print('------------------------------------------------------------------------')
      if response.message.tool_calls:
        for tc in response.message.tool_calls:
          if tc.function.name in available_functions:
            if print_flag:
              print(f"Calling {tc.function.name} with arguments {tc.function.arguments}")
            result = available_functions[tc.function.name](**tc.function.arguments)
            if print_flag:
              print(f"Result: {result}")
              print('------------------------------------------------------------------------')
            # add the tool result to the messages
            messages.append({'role': 'tool', 'tool_name': tc.function.name, 'content': str(result)})
      else:
        # end the loop when there are no more tool calls
        break

  return '', None, messages[-1]['content']

# Vault lives at the repo root (modrag runs from code/, so ../vault).
VAULT_DIR = '../vault'

def handle_memory_prompt(prompt):
  '''
    Intercepts the memory/recall keywords before the prompt is sent to the
    model. Returns (handled, output) where `handled` is True if the prompt was
    a memory keyword (and should NOT be forwarded to the model) and `output`
    is the markdown string to print to the user, or None.

    Keywords:
      memory | save memory | remember   -> write this session to the vault
      recall                              -> list saved sessions
      recall last | recall <date>         -> load a saved session into context
    '''
  global session_start_time
  raw = prompt.strip()
  low = raw.lower()

  # --- write ---
  if low in ('memory', 'save memory', 'remember'):
    status = modrag_memory.save_session(messages, session_start_time, vault_dir=VAULT_DIR)
    # reset the window so the next memory only covers work done after now
    session_start_time = time.time()
    return True, status

  # --- read ---
  if low == 'recall':
    return True, modrag_memory.list_sessions(vault_dir=VAULT_DIR)
  if low.startswith('recall '):
    which = raw[len('recall '):].strip().lower() or 'last'
    body = modrag_memory.recall_session(vault_dir=VAULT_DIR, which=which)
    if body is None:
      return True, f'No saved session matched "{which}".'
    # inject the recalled summary as context so the next chat_turn knows it
    messages.append({'role': 'user',
                      'content': f'Here is the summary of a previous Modrag '
                                 f'session for your context:\n\n{body}'})
    return True, body

  return False, None

start_chat()
header_string = f'''
\033[1;36m****************************************\033[0m
\033[1;35m* _   _ _     ___ _                    *\033[0m
\033[1;36m*| | | (_)   |_ _( )_ __ ___           *\033[0m
\033[1;35m*| |_| | |    | ||/| '_ ` _ \          *\033[0m
\033[1;36m*|  _  | |_   | |  | | | | | |         *\033[0m
\033[1;35m*|_| |_|_( ) |___| |_| |_| |_|         *\033[0m
\033[1;36m*        |/                            *\033[0m
\033[38;5;208m* __  __       ____        _         _ *\033[0m
\033[1;35m*|  \/  | ___ |  _ \ _ __ / \   __ _| |*\033[0m
\033[1;36m*| |\/| |/ _ \| | | | '__/ _ \ / _` | |*\033[0m
\033[1;35m*| |  | | (_) | |_| | | / ___ \ (_| |_|*\033[0m
\033[38;5;208m*|_|  |_|\___/|____/|_|/_/   \_\__, (_)*\033[0m
\033[1;36m*                              |___/   *\033[0m
\033[38;5;208m* A CafChem project!                   *\033[0m
\033[1;36m****************************************\033[0m
\033[1;35mThe MOdular DRug design AGent!\033[0m
\033[1;36mA command-line interface (CLI) for drug\033[0m
\033[1;35mdiscovery and molecular design.\033[0m
\033[38;5;208mType `memory` to save this session, `recall` to revisit one, `quit` to exit.\033[0m
\033[0m'''


print(header_string)

next_prompt = input("\033[1;36mWhat can I help with today? > \033[0m")
print('')
if next_prompt == 'quit':
  print("\033[1;35mResponse > \033[0mGoodbye!")
else:
  handled, mem_output = handle_memory_prompt(next_prompt)
  if handled:
    console.print(Markdown(to_unicode_markdown(mem_output or '')))
  else:
    # Get image modification time before
    img_path = '../images/chat_image.png'
    img_mtime_before = os.path.getmtime(img_path) if os.path.exists(img_path) else None

    start_time = time.time()
    _, _, response_content = chat_turn(next_prompt)
    end_time = time.time()

    # Check if image was modified
    img_mtime_after = os.path.getmtime(img_path) if os.path.exists(img_path) else None

    time_for_inf = (end_time - start_time) / 60
    print(f"\033[1;35mResponse {time_for_inf:.2f}m > \033[0m")
    console.print(Markdown(to_unicode_markdown(response_content)))

    if img_mtime_after and img_mtime_before != img_mtime_after:
      print(f"\033[38;5;208mNote: Image available at {img_path}\033[0m")

while next_prompt != 'quit':
  print('')
  next_prompt = input("\033[1;36mWhat else can I help with? > \033[0m")
  print('')
  if next_prompt == 'quit':
    print("\033[1;35mResponse > \033[0mGoodbye!")
    break
  handled, mem_output = handle_memory_prompt(next_prompt)
  if handled:
    console.print(Markdown(to_unicode_markdown(mem_output or '')))
    continue
  # Get image modification time before
  img_path = '../images/chat_image.png'
  img_mtime_before = os.path.getmtime(img_path) if os.path.exists(img_path) else None

  start_time = time.time()
  _, _, response_content = chat_turn(next_prompt)
  end_time = time.time()

  # Check if image was modified
  img_mtime_after = os.path.getmtime(img_path) if os.path.exists(img_path) else None

  time_for_inf = (end_time - start_time) / 60
  print('')
  print(f"\033[1;35mResponse {time_for_inf:.2f}m > \033[0m")
  console.print(Markdown(to_unicode_markdown(response_content)))

  if img_mtime_after and img_mtime_before != img_mtime_after:
    print(f"\033[38;5;208mNote: Image available at {img_path}\033[0m")
