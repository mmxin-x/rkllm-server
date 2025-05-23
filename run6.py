import ctypes
import os

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

    # Deduplicate paths while preserving order (for Python 3.7+)
    seen = set()
    unique_paths = []
    for path_item in paths:
        if path_item not in seen:
            seen.add(path_item)
            unique_paths.append(path_item)
    
    print(f"Searching for '{lib_name}' in paths: {unique_paths}") # Debug print

    for path in unique_paths:
        lib_path = os.path.join(path, lib_name)
        if os.path.exists(lib_path):
            print(f"Found library at: {lib_path}") # Debug print
            return lib_path
        else:
            print(f"Not found at: {lib_path}") # Debug print
            
    return None

# --- Load RKLLM Shared Library ---
# Adjust 'librkllmrt.so' if the name is different or provide a full path
LIB_RKLLM_PATH = find_library('librkllmrt.so')
if LIB_RKLLM_PATH is None:
    # Fallback if not found in common paths, user might need to set this manually
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
# LLMCallState (Page 2)
LLM_RUN_NORMAL = 0
LLM_RUN_FINISH = 1
LLM_RUN_WAITING = 2 # Not explicitly numbered, assuming sequence
LLM_RUN_ERROR = 3   # Not explicitly numbered, assuming sequence

# RKLLMInputType (Page 8)
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_TOKEN = 1
RKLLM_INPUT_EMBED = 2
RKLLM_INPUT_MULTIMODAL = 3

# RKLLMInferMode (Page 10)
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_GET_LOGITS = 1


# --- Structure Definitions (inferred from PDF) ---

class RKLLMExtendParam(ctypes.Structure):
    """ Corresponds to RKLLMExtendParam structure (Page 6) """
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
    ]

class RKLLMParam(ctypes.Structure):
    """ Corresponds to RKLLMParam structure (Page 4-7) """
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
        ("img_start", ctypes.c_char_p), # For multimodal
        ("img_end", ctypes.c_char_p),   # For multimodal
        ("img_content", ctypes.c_char_p),# For multimodal
        ("extend_param", RKLLMExtendParam),
        ("n_keep", ctypes.c_int32), # Assuming int32 based on other int types
    ]

class RKLLMResult(ctypes.Structure):
    """ Corresponds to RKLLMResult structure (Page 2-3) """
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32), 
    ]

class _RKLLMInputUnion(ctypes.Union):
    """ Union part of RKLLMInput (Page 7-8) """
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
    ]

class RKLLMInput(ctypes.Structure):
    """ Corresponds to RKLLMInput structure (Page 7-8) """
    _anonymous_ = ("_input_data",) 
    _fields_ = [
        ("input_type", ctypes.c_int), 
        ("_input_data", _RKLLMInputUnion),
    ]

class RKLLMInferParam(ctypes.Structure):
    """ Corresponds to RKLLMInferParam structure (Page 10) """
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
    result = result_ptr.contents
    if state == LLM_RUN_NORMAL:
        if result.text:
            try:
                print(result.text.decode('utf-8', errors='replace'), end="", flush=True)
            except Exception as e:
                print(f"[Callback Error decoding text: {e}]", end="", flush=True)
    elif state == LLM_RUN_FINISH:
        print("\n<LLM_RUN_FINISH>")
    elif state == LLM_RUN_ERROR:
        print("\n<LLM_RUN_ERROR>")
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


# --- Main Inference Logic ---
def run_inference():
    # Enable RKLLM Logging (Level 1 for performance, 2 for detailed)
    # This should be set before library initialization ideally.
    os.environ['RKLLM_LOG_LEVEL'] = '1' 
    print(f"RKLLM_LOG_LEVEL set to: {os.environ['RKLLM_LOG_LEVEL']}")

    print("Starting RKLLM Inference Script...")

    llm_handle = LLMHandle()

    print("Creating default parameters...")
    param = rkllm.rkllm_createDefaultParam()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # New model path as requested by user
    model_relative_path = "../model/Qwen3-0.6B-w8a8-opt1-hybrid1-npu3.rkllm"
    
    # Construct the absolute path to the model
    # os.path.join will correctly handle the '../' part if script_dir is an absolute path
    model_abs_path = os.path.join(script_dir, model_relative_path)
    
    # Normalize the path to resolve '..' and get the canonical path
    # This makes the path cleaner and ensures it's absolute
    model_canonical_path = os.path.normpath(model_abs_path)

    if os.path.exists(model_canonical_path):
        param.model_path = model_canonical_path.encode('utf-8')
    else:
        print(f"Error: Model file not found at '{model_canonical_path}' (derived from relative path '{model_relative_path}')")
        print(f"Please ensure the model exists at the specified relative path from the script directory.")
        return
    
    print(f"Using model: {param.model_path.decode()}")

    print("Initializing RKLLM model...")
    ret = rkllm.rkllm_init(ctypes.byref(llm_handle), ctypes.byref(param), python_llm_callback)
    if ret != 0:
        print(f"Error: rkllm_init failed with code {ret}")
        return
    print("RKLLM model initialized successfully.")

    prompt_prefix = b"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    user_query = b"What is Rockchip?"
    prompt_postfix = b"<|im_end|>\n<|im_start|>assistant\n"
    
    full_prompt_str = prompt_prefix + user_query + prompt_postfix
    
    rkllm_input = RKLLMInput()
    rkllm_input.input_type = RKLLM_INPUT_PROMPT
    rkllm_input.prompt_input = ctypes.c_char_p(full_prompt_str) 

    rkllm_infer_params = RKLLMInferParam()
    rkllm_infer_params.mode = RKLLM_INFER_GENERATE
    rkllm_infer_params.keep_history = 0 

    print(f"\nSending prompt: {full_prompt_str.decode('utf-8', errors='replace')}")
    print("Assistant's Response:")
    
    ret = rkllm.rkllm_run(llm_handle, ctypes.byref(rkllm_input), ctypes.byref(rkllm_infer_params), None) 
    if ret != 0:
        print(f"\nError: rkllm_run failed with code {ret}")

    print("Destroying RKLLM model...")
    ret_destroy = rkllm.rkllm_destroy(llm_handle)
    if ret_destroy != 0:
        print(f"Error: rkllm_destroy failed with code {ret_destroy}")
    else:
        print("RKLLM model destroyed successfully.")

if __name__ == "__main__":
    run_inference()
