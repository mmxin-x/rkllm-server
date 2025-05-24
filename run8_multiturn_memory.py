import ctypes
import os

# --- Global list to accumulate assistant's response parts ---
current_assistant_response_parts = []

# --- Helper to find the library ---
def find_library(lib_name):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Common library paths
    paths = [
        '.', # Current working directory
        script_dir, # Directory of the script itself
        os.path.join(script_dir, 'aarch64'), # aarch64 folder sibling to script's parent
        os.path.join(script_dir, 'lib'), # if lib is in a subdir of the script's dir
        '/usr/lib',
        '/usr/local/lib',
    ]
    # Add paths from LD_LIBRARY_PATH
    if 'LD_LIBRARY_PATH' in os.environ:
        paths.extend(os.environ['LD_LIBRARY_PATH'].split(os.pathsep))

    # Deduplicate paths while preserving order
    seen = set()
    unique_paths = []
    for path_item in paths:
        if path_item not in seen:
            seen.add(path_item)
            unique_paths.append(path_item)
    
    print(f"Searching for '{lib_name}' in paths: {unique_paths}") 

    for path in unique_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            print(f"Found library at: {lib_path}") 
            return lib_path
        else:
            print(f"Not found at: {lib_path}") 
            
    return None

# --- Load RKLLM Shared Library ---
LIB_RKLLM_PATH = find_library('librkllmrt.so')
if LIB_RKLLM_PATH is None:
    LIB_RKLLM_PATH = 'librkllmrt.so' 
    print(f"Warning: '{LIB_RKLLM_PATH}' could not be found automatically. Trying to load directly.")
    print(f"Ensure '{LIB_RKLLM_PATH}' is in your LD_LIBRARY_PATH or provide an absolute path if loading fails.")

try:
    rkllm = ctypes.CDLL(LIB_RKLLM_PATH)
    print(f"Successfully loaded RKLLM library from '{LIB_RKLLM_PATH}'")
except OSError as e:
    print(f"Error loading RKLLM library from '{LIB_RKLLM_PATH}': {e}")
    print("Please ensure the library path is correct, it's for aarch64 architecture, and all dependencies are met.")
    exit(1)

# --- Constants from rkllm.h (inferred from PDF) ---
LLM_RUN_NORMAL = 0
LLM_RUN_FINISH = 1
LLM_RUN_WAITING = 2 
LLM_RUN_ERROR = 3   

RKLLM_INPUT_PROMPT = 0
# Other input types not used in this version
# RKLLM_INPUT_TOKEN = 1
# RKLLM_INPUT_EMBED = 2
# RKLLM_INPUT_MULTIMODAL = 3

RKLLM_INFER_GENERATE = 0
# RKLLM_INFER_GET_LOGITS = 1


# --- Structure Definitions (inferred from PDF) ---
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("num_npu_core", ctypes.c_int32),
        ("use_gpu", ctypes.c_bool),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p), 
        ("img_end", ctypes.c_char_p),   
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
        ("n_keep", ctypes.c_int32), 
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32), 
    ]

class _RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
    ]

class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("_input_data",) 
    _fields_ = [
        ("input_type", ctypes.c_int), 
        ("_input_data", _RKLLMInputUnion),
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int), 
        ("lora_params", ctypes.c_void_p), 
        ("prompt_cache_params", ctypes.c_void_p), 
        ("keep_history", ctypes.c_int), 
    ]

LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

@LLMResultCallback
def python_llm_callback(result_ptr, userdata, state):
    """ Python implementation of the LLMResultCallback """
    global current_assistant_response_parts

    if not result_ptr:
        print(f"\n[Callback Warning] Received NULL result_ptr. State: {state}")
        if state == LLM_RUN_ERROR:
            print("[Callback Info] This often indicates an error within the RKLLM library during the current inference step.")
        return 

    result = result_ptr.contents 
    if state == LLM_RUN_NORMAL:
        if result.text:
            try:
                decoded_text = result.text.decode('utf-8', errors='replace')
                print(decoded_text, end="", flush=True)
                current_assistant_response_parts.append(decoded_text)
            except Exception as e:
                print(f"[Callback Error decoding text: {e}]", end="", flush=True)
    elif state == LLM_RUN_FINISH:
        print("\n<LLM_RUN_FINISH>") 
    elif state == LLM_RUN_ERROR:
        print("\n<LLM_RUN_ERROR>")
        if result.text: 
             try:
                print(f"Error details from result: {result.text.decode('utf-8', errors='replace')}", end="", flush=True)
             except:
                pass 
    elif state == LLM_RUN_WAITING:
        pass 


# --- RKLLM API Function Prototypes ---
rkllm.rkllm_createDefaultParam.restype = RKLLMParam
rkllm.rkllm_createDefaultParam.argtypes = []

LLMHandle = ctypes.c_void_p 
rkllm.rkllm_init.restype = ctypes.c_int
rkllm.rkllm_init.argtypes = [ctypes.POINTER(LLMHandle), ctypes.POINTER(RKLLMParam), LLMResultCallback]

rkllm.rkllm_run.restype = ctypes.c_int
rkllm.rkllm_run.argtypes = [LLMHandle, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]

rkllm.rkllm_destroy.restype = ctypes.c_int
rkllm.rkllm_destroy.argtypes = [LLMHandle]

# --- ADDED: rkllm_clear_kv_cache prototype ---
# int rkllm_clear_kv_cache(LLMHandle handle, int keep_system_prompt); (Page 18 of PDF)
rkllm.rkllm_clear_kv_cache.restype = ctypes.c_int
rkllm.rkllm_clear_kv_cache.argtypes = [LLMHandle, ctypes.c_int]
# --- END OF ADDED PROTOTYPE ---


# --- Main Inference Logic ---
def run_inference():
    global current_assistant_response_parts
    os.environ['RKLLM_LOG_LEVEL'] = '1' 
    print(f"RKLLM_LOG_LEVEL set to: {os.environ['RKLLM_LOG_LEVEL']}")
    print("Starting RKLLM Inference Script for Continuous Conversation...")

    llm_handle = LLMHandle()

    print("Creating default parameters...")
    param = rkllm.rkllm_createDefaultParam()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_relative_path = "../model/Gemma3-1B-w8a8-opt1.rkllm"
    model_abs_path = os.path.join(script_dir, model_relative_path)
    model_canonical_path = os.path.normpath(model_abs_path)

    if os.path.exists(model_canonical_path):
        param.model_path = model_canonical_path.encode('utf-8')
    else:
        print(f"Error: Model file not found at '{model_canonical_path}'")
        return
    
    param.use_gpu = True 
    param.max_context_len = 1000 
    param.n_keep = 32 
    
    if param.max_new_tokens == 0:
        param.max_new_tokens = 256 
        print(f"Set max_new_tokens to {param.max_new_tokens} (default was 0)")

    print(f"Using model: {param.model_path.decode()}")
    print(f"GPU usage for prefill acceleration: {'Enabled' if param.use_gpu else 'Disabled'}")
    print(f"Max context length (overall KV cache): {param.max_context_len}")
    print(f"Max new tokens per turn: {param.max_new_tokens}")
    print(f"N_keep (tokens to keep from start on cache clear): {param.n_keep}")

    print("Initializing RKLLM model...")
    ret = rkllm.rkllm_init(ctypes.byref(llm_handle), ctypes.byref(param), python_llm_callback)
    if ret != 0:
        print(f"Error: rkllm_init failed with code {ret}")
        return
    print("RKLLM model initialized successfully.")

    SYS_PROMPT = b"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    USER_TURN_START = b"<|im_start|>user\n"
    USER_TURN_END = b"<|im_end|>\n"
    ASSISTANT_TURN_START = b"<|im_start|>assistant\n"
    ASSISTANT_TURN_END = b"<|im_end|>\n"

    conversation_history_bytes = SYS_PROMPT

    rkllm_infer_params = RKLLMInferParam()
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE
    rkllm_infer_params.keep_history = 1 

    # --- Heuristic limit for prompt token length based on observed library error ---
    # The library error "max context: 512" for the prompt seems to be a hard limit for the prompt part.
    EFFECTIVE_PROMPT_TOKEN_LIMIT = 500 # Giving a small buffer from 512

    print("\nStarting chat. Type 'quit' or 'exit' to end.")
    while True:
        try:
            user_input_str = input("You: ")
        except EOFError: 
            print("\nExiting due to EOF...")
            break
        if user_input_str.lower() in ["quit", "exit"]:
            print("Exiting...")
            break
        if not user_input_str.strip(): 
            continue

        current_assistant_response_parts.clear() 

        current_user_turn_bytes = USER_TURN_START + user_input_str.encode('utf-8') + USER_TURN_END
        prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
        
        # --- Proactive KV Cache Clearing based on EFFECTIVE_PROMPT_TOKEN_LIMIT ---
        estimated_prompt_tokens = len(prompt_for_llm) / 3 # Rough estimate (bytes to tokens)
        
        if estimated_prompt_tokens > EFFECTIVE_PROMPT_TOKEN_LIMIT:
            print(f"\n[Context Management] Estimated prompt tokens ({int(estimated_prompt_tokens)}) exceeds effective limit ({EFFECTIVE_PROMPT_TOKEN_LIMIT}). Clearing KV cache.")
            
            clear_ret = rkllm.rkllm_clear_kv_cache(llm_handle, 1) # 1 = keep system prompt
            if clear_ret == 0:
                print("[Context Management] KV cache cleared (system prompt kept via n_keep/flag).")
                conversation_history_bytes = SYS_PROMPT # Reset history to system prompt
                # Rebuild prompt for the current turn with reset history
                prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
                print(f"[Context Management] Conversation history reset. New prompt length estimate: {int(len(prompt_for_llm)/3)} tokens.")
            else:
                print(f"[Context Management] Error clearing KV cache (code: {clear_ret}). Proceeding with potentially long prompt.")
        # --- End of Proactive Clearing ---

        # General warning if total context (prompt + potential generation) might exceed overall max_context_len
        # This is a secondary check, the primary one is for the prompt length itself.
        elif estimated_prompt_tokens > (param.max_context_len - param.max_new_tokens):
             print(f"\n[Warning] Estimated prompt tokens ({int(estimated_prompt_tokens)}) + max_new_tokens ({param.max_new_tokens}) may exceed total max_context_len ({param.max_context_len}).")


        print("Assistant: ", end="") 

        rkllm_input_obj = RKLLMInput()
        rkllm_input_obj.input_type = RKLLM_INPUT_PROMPT
        rkllm_input_obj.prompt_input = ctypes.c_char_p(prompt_for_llm)

        ret = rkllm.rkllm_run(llm_handle, ctypes.byref(rkllm_input_obj), ctypes.byref(rkllm_infer_params), None) 
        
        if ret != 0:
            print(f"\nError: rkllm_run failed with code {ret}. Conversation might be affected.")
            # If rkllm_run fails, especially after a clear, it might be that the single new turn is still too long.
            # Or another unrelated error.
            continue
        
        full_assistant_response_str = "".join(current_assistant_response_parts)
        
        if ret == 0: 
            # Append current user turn and assistant's full response to history
            new_dialogue_segment = current_user_turn_bytes + ASSISTANT_TURN_START + \
                                   full_assistant_response_str.encode('utf-8') + ASSISTANT_TURN_END
            conversation_history_bytes += new_dialogue_segment     

    print("\nDestroying RKLLM model...")
    ret_destroy = rkllm.rkllm_destroy(llm_handle)
    if ret_destroy != 0:
        print(f"Error: rkllm_destroy failed with code {ret_destroy}")
    else:
        print("RKLLM model destroyed successfully.")

if __name__ == "__main__":
    run_inference()
