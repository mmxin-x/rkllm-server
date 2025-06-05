"""
RKLLM Runtime Interface Module

This module provides the interface to the RKLLM native library (librkllmrt.so).
It handles loading the library, initializing the model, and defining the necessary
ctypes structures for communication with the native code.
"""

import ctypes
import os
import threading
import queue
from typing import Optional, Callable, Dict, Any

from ..config import (
    LIBRARY_PATH, MODEL_PATH, MAX_CONTEXT_LENGTH, MAX_NEW_TOKENS, 
    N_KEEP, USE_GPU, IS_ASYNC, LOG_LEVEL
)

# Define C structures for RKLLM library
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32)
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
        ("n_keep", ctypes.c_int32)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32)
    ]

class _RKLLMInputUnion(ctypes.Union):
    _fields_ = [("prompt_input", ctypes.c_char_p)]

class RKLLMInput(ctypes.Structure):
    _anonymous_ = ("_input_data",)
    _fields_ = [
        ("input_type", ctypes.c_int),
        ("_input_data", _RKLLMInputUnion)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int)
    ]

# Define callback type
LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)

# Constants
RKLLM_INPUT_PROMPT = 0
RKLLM_INPUT_IMAGE = 1
RKLLM_INFER_GENERATE = 0
RKLLM_INFER_EMBEDDING = 1
RKLLM_RESULT_OK = 0
RKLLM_RESULT_SINGLE = 0
RKLLM_RESULT_FINISH = 1
RKLLM_RESULT_ERROR = 2

class RKLLMRuntime:
    """
    RKLLM Runtime Interface
    
    This class handles the initialization and interaction with the RKLLM native library.
    """
    def __init__(self):
        self.library = None
        self.handle = ctypes.c_void_p()
        self.params = None
        self.lock = threading.RLock()
        self.initialized = False
        self.token_queue = queue.Queue()
        self.callback = None
    
    def load_library(self, library_path: str = LIBRARY_PATH) -> bool:
        """
        Load the RKLLM native library.
        
        Args:
            library_path: Path to the RKLLM native library (default from config)
            
        Returns:
            bool: True if library was loaded successfully, False otherwise
        """
        with self.lock:
            if self.library:
                return True
                
            try:
                print(f"Loading RKLLM library from '{library_path}'...")
                self.library = ctypes.CDLL(library_path)
                print(f"Loaded RKLLM library from '{library_path}'")
                
                # Define function signatures
                self.library.rkllm_createDefaultParam.restype = RKLLMParam
                self.library.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback]
                self.library.rkllm_init.restype = ctypes.c_int
                self.library.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
                self.library.rkllm_run.restype = ctypes.c_int
                self.library.rkllm_destroy.argtypes = [ctypes.c_void_p]
                self.library.rkllm_destroy.restype = ctypes.c_int
                self.library.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int]
                self.library.rkllm_clear_kv_cache.restype = ctypes.c_int
                
                return True
            except Exception as e:
                print(f"Error loading RKLLM library: {e}")
                return False
                
    def _callback_handler(self, result_ptr, userdata, state):
        """Internal callback handler for RKLLM library"""
        if not result_ptr:
            self.token_queue.put(None)
            return
            
        result = result_ptr.contents
        if state == LLM_RUN_NORMAL:
            if result.text:
                try:
                    decoded_text = result.text.decode('utf-8', errors='replace')
                    self.token_queue.put((decoded_text, userdata, state))
                except Exception as e:
                    print(f"[Callback Error] Decoding text: {e}")
                    self.token_queue.put(None)
        elif state == LLM_RUN_FINISH:
            self.token_queue.put(("__LLM_RUN_FINISH__", userdata, state))
        elif state == LLM_RUN_ERROR:
            self.token_queue.put(None)
    
    def initialize_model(self, 
                         model_path: str = MODEL_PATH,
                         max_context_len: int = MAX_CONTEXT_LENGTH,
                         max_new_tokens: int = MAX_NEW_TOKENS,
                         n_keep: int = N_KEEP,
                         use_gpu: bool = USE_GPU,
                         is_async: bool = IS_ASYNC) -> bool:
        """
        Initialize the RKLLM model.
        
        Args:
            model_path: Path to the RKLLM model
            max_context_len: Maximum context length
            max_new_tokens: Maximum number of new tokens to generate
            n_keep: Number of tokens to keep in history
            use_gpu: Whether to use GPU
            is_async: Whether to run inference asynchronously
            
        Returns:
            bool: True if model was initialized successfully, False otherwise
        """
        if not self.library:
            print("Error: RKLLM library not loaded")
            return False
            
        with self.lock:
            try:
                print(f"Initializing model with parameters: ")
                print(f"  Path: {model_path}")
                print(f"  Max Context: {max_context_len}")
                print(f"  Max New Tokens: {max_new_tokens}")
                print(f"  N_Keep: {n_keep}")
                print(f"  Use GPU: {use_gpu}")
                print(f"  Is Async (Library Flag): {is_async}")
                
                # Create parameter structure from default
                self.params = self.library.rkllm_createDefaultParam()
                self.params.model_path = model_path.encode('utf-8')
                self.params.max_context_len = max_context_len
                # Only set max_new_tokens if provided
                if self.params.max_new_tokens == 0 and max_new_tokens > 0:
                    self.params.max_new_tokens = max_new_tokens
                self.params.n_keep = n_keep
                self.params.use_gpu = use_gpu
                self.params.is_async = is_async
                
                # Set default values for other parameters
                self.params.num_npu_core = 3  # RK3588 has 3 NPU cores
                self.params.top_k = 30000
                self.params.top_p = 0.9
                self.params.temperature = 0.8
                self.params.repeat_penalty = 1.1
                self.params.frequency_penalty = 0.0
                self.params.mirostat = 0
                self.params.mirostat_tau = 5.0
                self.params.mirostat_eta = 0.1
                self.params.skip_special_token = True
                self.params.img_start = ctypes.c_char_p()
                self.params.img_end = ctypes.c_char_p()
                self.params.img_content = ctypes.c_char_p()
                
                # Initialize extend params
                extend_param = RKLLMExtendParam()
                extend_param.base_domain_id = 0
                extend_param.embed_flash = 1
                extend_param.enabled_cpus_num = 4
                extend_param.enabled_cpus_mask = 0xF0  # Use cores 4,5,6,7
                self.params.extend_param = extend_param
                
                # Setup callback
                callback_func = LLMResultCallback(self._callback_handler)
                self.callback = callback_func  # Save reference to prevent garbage collection
                
                # Initialize RKLLM with the proper function signature
                ret = self.library.rkllm_init(ctypes.byref(self.handle), ctypes.byref(self.params), callback_func)
                
                if ret == 0:
                    self.initialized = True
                    print("RKLLM model initialized.")
                    return True
                else:
                    print(f"Failed to initialize RKLLM model. Error code: {ret}")
                    return False
                    
            except Exception as e:
                print(f"Error initializing RKLLM model: {e}")
                return False
    
    def run_inference(self, 
                      prompt: str,
                      callback: Callable, 
                      userdata: Any,
                      keep_history: bool = True) -> int:
        """
        Run inference with the RKLLM model.
        
        Args:
            prompt: Input prompt
            callback: Callback function to receive results
            userdata: User data to pass to callback
            keep_history: Whether to keep history
            
        Returns:
            int: Return code from RKLLM run
        """
        if not self.initialized:
            print("Error: RKLLM model not initialized")
            return -1
            
        with self.lock:
            try:
                # Clear token queue
                while not self.token_queue.empty():
                    self.token_queue.get_nowait()
                
                # Create input structure
                input_obj = RKLLMInput()
                input_obj.input_type = RKLLM_INPUT_PROMPT
                input_obj.prompt_input = prompt.encode('utf-8')
                
                # Create inference parameters
                infer_params = RKLLMInferParam()
                infer_params.mode = RKLLM_INFER_GENERATE
                infer_params.lora_params = None
                infer_params.prompt_cache_params = None
                infer_params.keep_history = 1 if keep_history else 0
                
                # Start callback thread
                thread = threading.Thread(
                    target=self._callback_worker,
                    args=(callback, userdata),
                    daemon=True
                )
                thread.start()
                
                # Run inference
                ret = self.library.rkllm_run(
                    self.handle,
                    ctypes.byref(input_obj),
                    ctypes.byref(infer_params),
                    userdata
                )
                
                return ret
                
            except Exception as e:
                print(f"Error running RKLLM inference: {e}")
                return -1
                
    def _callback_worker(self, external_callback, userdata):
        """Worker thread to process tokens from the queue"""
        while True:
            try:
                # Get next token with timeout
                item = self.token_queue.get(timeout=0.1)
                
                if item is None:
                    # Error or completion
                    if external_callback:
                        external_callback(None, userdata, LLM_RUN_ERROR)
                    break
                    
                token, token_userdata, state = item
                
                if token == "__LLM_RUN_FINISH__":
                    # Finished normally
                    if external_callback:
                        external_callback(None, userdata, LLM_RUN_FINISH)
                    break
                    
                # Normal token
                if external_callback:
                    external_callback(token, userdata, state)
                    
            except queue.Empty:
                # No tokens yet, continue waiting
                continue
            except Exception as e:
                print(f"Error in callback worker: {e}")
                if external_callback:
                    try:
                        external_callback(None, userdata, LLM_RUN_ERROR)
                    except:
                        pass
                break
    
    def clear_kv_cache(self) -> int:
        """
        Clear the KV cache.
        
        Returns:
            int: Return code from RKLLM clear_kv_cache
        """
        if not self.initialized:
            print("Error: RKLLM model not initialized")
            return -1
            
        with self.lock:
            try:
                return self.library.rkllm_clear_kv_cache(self.handle)
            except Exception as e:
                print(f"Error clearing KV cache: {e}")
                return -1
    
    def cleanup(self) -> bool:
        """
        Clean up RKLLM resources.
        
        Returns:
            bool: True if cleanup was successful, False otherwise
        """
        if not self.initialized:
            return True
            
        with self.lock:
            try:
                if self.handle:
                    result = self.library.rkllm_destroy(self.handle)
                    self.handle = None
                    self.initialized = False
                    return result == RKLLM_RESULT_OK
                return True
            except Exception as e:
                print(f"Error cleaning up RKLLM resources: {e}")
                return False

# Create a global instance
runtime = RKLLMRuntime()

def initialize() -> bool:
    """
    Initialize the RKLLM runtime.
    
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    if runtime.load_library():
        return runtime.initialize_model()
    return False

def cleanup() -> bool:
    """
    Clean up RKLLM runtime resources.
    
    Returns:
        bool: True if cleanup was successful, False otherwise
    """
    return runtime.cleanup()
