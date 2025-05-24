import ctypes
import os
import time
import json
import uuid
import threading
import queue # For thread-safe queue
from flask import Flask, request, Response, jsonify, stream_with_context

# --- Global State & Configuration ---
rkllm_lib = None
llm_handle = ctypes.c_void_p()
rkllm_params_global = None
model_initialized = False
current_assistant_response_parts = [] 

rkllm_lock = threading.Lock() # Lock for critical RKLLM sections and shared state

# --- RKLLM Constants & Structures ---
LLM_RUN_NORMAL = 0
LLM_RUN_FINISH = 1
LLM_RUN_WAITING = 2
LLM_RUN_ERROR = 3
RKLLM_INPUT_PROMPT = 0
RKLLM_INFER_GENERATE = 0

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [("base_domain_id", ctypes.c_int32), ("embed_flash", ctypes.c_int8), ("enabled_cpus_num", ctypes.c_int8), ("enabled_cpus_mask", ctypes.c_uint32)]
class RKLLMParam(ctypes.Structure):
    _fields_ = [("model_path", ctypes.c_char_p), ("num_npu_core", ctypes.c_int32), ("use_gpu", ctypes.c_bool), ("max_context_len", ctypes.c_int32), ("max_new_tokens", ctypes.c_int32), ("top_k", ctypes.c_int32), ("top_p", ctypes.c_float), ("temperature", ctypes.c_float), ("repeat_penalty", ctypes.c_float), ("frequency_penalty", ctypes.c_float), ("mirostat", ctypes.c_int32), ("mirostat_tau", ctypes.c_float), ("mirostat_eta", ctypes.c_float), ("skip_special_token", ctypes.c_bool), ("is_async", ctypes.c_bool), ("img_start", ctypes.c_char_p), ("img_end", ctypes.c_char_p), ("img_content", ctypes.c_char_p), ("extend_param", RKLLMExtendParam), ("n_keep", ctypes.c_int32)]
class RKLLMResult(ctypes.Structure): _fields_ = [("text", ctypes.c_char_p), ("token_id", ctypes.c_int32)]
class _RKLLMInputUnion(ctypes.Union): _fields_ = [("prompt_input", ctypes.c_char_p)]
class RKLLMInput(ctypes.Structure): _anonymous_ = ("_input_data",); _fields_ = [("input_type", ctypes.c_int), ("_input_data", _RKLLMInputUnion)]
class RKLLMInferParam(ctypes.Structure): _fields_ = [("mode", ctypes.c_int), ("lora_params", ctypes.c_void_p), ("prompt_cache_params", ctypes.c_void_p), ("keep_history", ctypes.c_int)]
LLMResultCallback = ctypes.CFUNCTYPE(None, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)








stream_token_queue = queue.Queue()

@LLMResultCallback
def api_llm_callback(result_ptr, userdata, state):
    global current_assistant_response_parts, stream_token_queue
    
    callback_thread_id = threading.get_ident()
    print(f"[Callback {callback_thread_id}] Received state: {state}. result_ptr is valid: {bool(result_ptr)}. Qsize before put: {stream_token_queue.qsize()}")

    if not result_ptr:
        print(f"[Callback {callback_thread_id}] CRITICAL: result_ptr is NULL. State was: {state}. Putting None on queue.")
        stream_token_queue.put(None) 
        return

    result = result_ptr.contents
    # print(f"[Callback {callback_thread_id}] result.token_id: {result.token_id}") # Optional

    if state == LLM_RUN_NORMAL:
        if result.text: # Check if the C char* pointer itself is not NULL
            try:
                # For deeper debugging, you could print the raw memory if suspecting non-string data
                # raw_bytes = ctypes.string_at(result.text) 
                # print(f"[Callback {callback_thread_id}] Raw bytes length: {len(raw_bytes)}, content: {raw_bytes[:50]}")
                
                decoded_text = result.text.decode('utf-8', errors='replace')
                if decoded_text.strip(): # Only process if there's actual content after decoding and stripping
                    print(f"[Callback {callback_thread_id}] NORMAL, Decoded Text: '{decoded_text}'")
                    current_assistant_response_parts.append(decoded_text) 
                    stream_token_queue.put(decoded_text)
                else:
                    print(f"[Callback {callback_thread_id}] NORMAL, but decoded_text is empty or whitespace after strip. Not queuing.")
            except Exception as e:
                print(f"[Callback {callback_thread_id}] Error decoding text: {e}. Putting None on queue.")
                stream_token_queue.put(None) # Signal error
        else:
            print(f"[Callback {callback_thread_id}] NORMAL, but result.text (C char*) is NULL. Not queuing text.")
            # If result.text is NULL in NORMAL state, it implies no token was generated for this step.
            # We should not put None on the queue here, as it might be an intermediate step.
            # The FINISH or ERROR signal will handle termination.
            
    elif state == LLM_RUN_FINISH:
        print(f"[Callback {callback_thread_id}] FINISH signaled. Putting __LLM_RUN_FINISH__ on queue.")
        stream_token_queue.put("__LLM_RUN_FINISH__") 
    elif state == LLM_RUN_ERROR:
        print(f"[Callback {callback_thread_id}] ERROR signaled. Putting None on queue.")
        stream_token_queue.put(None) 
    elif state == LLM_RUN_WAITING: # Usually for incomplete UTF-8, library handles internally.
        print(f"[Callback {callback_thread_id}] WAITING signaled. (Typically informational)")
    else:
        print(f"[Callback {callback_thread_id}] Unknown state: {state}. Putting None on queue as a precaution.")
        stream_token_queue.put(None)


def find_library_path(lib_name):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    paths_to_check = ['.', script_dir, os.path.join(script_dir, 'aarch64'), os.path.join(script_dir, 'lib'), '/usr/lib', '/usr/local/lib']
    if 'LD_LIBRARY_PATH' in os.environ: paths_to_check.extend(os.environ['LD_LIBRARY_PATH'].split(os.pathsep))
    for p_item in dict.fromkeys(paths_to_check):
        _lp = os.path.join(p_item, lib_name)
        if os.path.exists(_lp): return _lp
    return None

def init_rkllm_model():
    global rkllm_lib, llm_handle, rkllm_params_global, model_initialized
    if model_initialized: return True
    lib_path = find_library_path('librkllmrt.so')
    if not lib_path: print("Error: librkllmrt.so not found."); return False
    try: rkllm_lib = ctypes.CDLL(lib_path); print(f"Loaded RKLLM library from '{lib_path}'")
    except OSError as e: print(f"Error loading RKLLM library: {e}"); return False

    rkllm_lib.rkllm_createDefaultParam.restype = RKLLMParam
    rkllm_lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), LLMResultCallback]; rkllm_lib.rkllm_init.restype = ctypes.c_int
    rkllm_lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]; rkllm_lib.rkllm_run.restype = ctypes.c_int
    rkllm_lib.rkllm_destroy.argtypes = [ctypes.c_void_p]; rkllm_lib.rkllm_destroy.restype = ctypes.c_int
    rkllm_lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int]; rkllm_lib.rkllm_clear_kv_cache.restype = ctypes.c_int
    
    rkllm_params_global = rkllm_lib.rkllm_createDefaultParam()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_relative_path = "../model/Gemma3-1B-w8a8-opt1.rkllm"
    model_abs_path = os.path.join(script_dir, model_relative_path)
    model_canonical_path = os.path.normpath(model_abs_path)
    if not os.path.exists(model_canonical_path): print(f"Error: Model file not found: '{model_canonical_path}'"); return False
    
    rkllm_params_global.model_path = model_canonical_path.encode('utf-8')
    rkllm_params_global.use_gpu = False
    rkllm_params_global.max_context_len = 30000 
    rkllm_params_global.n_keep = 32
    if rkllm_params_global.max_new_tokens == 0: rkllm_params_global.max_new_tokens = 512
    
    rkllm_params_global.is_async = False # Using synchronous mode with worker thread for streaming
    
    os.environ['RKLLM_LOG_LEVEL'] = '1' 
    print(f"RKLLM_LOG_LEVEL: {os.environ['RKLLM_LOG_LEVEL']}")
    print(f"Initializing model with parameters: \n"
          f"  Path: {rkllm_params_global.model_path.decode()}\n"
          f"  Max Context: {rkllm_params_global.max_context_len}\n"
          f"  Max New Tokens: {rkllm_params_global.max_new_tokens}\n"
          f"  N_Keep: {rkllm_params_global.n_keep}\n"
          f"  Use GPU: {rkllm_params_global.use_gpu}\n"
          f"  Is Async (Library Flag): {rkllm_params_global.is_async}")
    
    ret = rkllm_lib.rkllm_init(ctypes.byref(llm_handle), ctypes.byref(rkllm_params_global), api_llm_callback)
    if ret != 0: print(f"Error: rkllm_init failed (code {ret})."); return False
    
    conversation_history_bytes = b""
    model_initialized = True
    print("RKLLM model initialized.")
    return True

app = Flask(__name__)

def format_openai_error_response(message, err_type="invalid_request_error", param=None, code=None):
    error_obj = {"error": {"message": message, "type": err_type}}
    if param: error_obj["error"]["param"] = param
    if code: error_obj["error"]["code"] = code
    return jsonify(error_obj)

def rkllm_inference_worker(handle, input_obj_ptr, infer_params_obj_ptr, worker_status_obj):
    thread_id = threading.get_ident()
    print(f"[Worker {thread_id}] Starting rkllm_run. Timestamp: {time.time()}")
    try:
        worker_status_obj.run_ret = rkllm_lib.rkllm_run(handle, input_obj_ptr, infer_params_obj_ptr, None)
        print(f"[Worker {thread_id}] rkllm_run finished with code: {worker_status_obj.run_ret}. Timestamp: {time.time()}")
        if worker_status_obj.run_ret != 0:
            print(f"[Worker {thread_id}] rkllm_run error, ensuring error signal in queue if not already sent by callback.")
            # Callback should ideally handle putting None on error. This is a fallback.
            # Check if queue is empty or last item isn't None or FINISH before putting another None.
            # This logic can be complex due to race conditions. Best if callback is robust.
            # For simplicity, we'll assume callback handles it or stream generator times out.
            # stream_token_queue.put(None) 
    except Exception as e:
        print(f"[Worker {thread_id}] Exception in rkllm_run: {e}")
        worker_status_obj.run_ret = -99 
        stream_token_queue.put(None) # Ensure error signal
    finally:
        worker_status_obj.finished = True
        # It's crucial that the FINISH or ERROR signal is put on the queue by the *callback*
        # to correctly terminate the stream generator. If rkllm_run finishes
        # but the last callback wasn't FINISH (e.g. due to error), the queue needs a terminator.
        # The callback logic should already handle this.

class WorkerStatus: 
    def __init__(self):
        self.run_ret = -1 
        self.finished = False

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions_handler():
    global current_assistant_response_parts, rkllm_params_global, llm_handle, rkllm_lib, stream_token_queue
    
    if not model_initialized: return format_openai_error_response("RKLLM model not initialized.", "api_error"), 500
    try: request_data = request.get_json()
    except Exception as e: return format_openai_error_response(f"Invalid JSON: {e}", "invalid_request_error"), 400
    if not request_data: return format_openai_error_response("Request body missing or not JSON.", "invalid_request_error"), 400

    messages = request_data.get("messages", [])
    stream_requested = request_data.get("stream", False)
    
    if not messages or messages[-1].get("role") != "user": return format_openai_error_response("Last message must be 'user'.", "invalid_request_error", param="messages"), 400
    last_user_message_content = messages[-1].get("content", "")
    if not last_user_message_content.strip(): return format_openai_error_response("User content is empty.", "invalid_request_error", param="messages.content"), 400

    with rkllm_lock: 
        current_user_turn_bytes = USER_TURN_START + last_user_message_content.encode('utf-8') + USER_TURN_END
        prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
        
        print(f"\n[API Handler DEBUG] Sending Prompt to RKLLM (first 500 bytes):\n'''\n{prompt_for_llm[:500].decode('utf-8', errors='replace')}...\n'''")
        if len(prompt_for_llm) > 500: print(f"[API Handler DEBUG] (Prompt is longer, total bytes: {len(prompt_for_llm)})")

        EFFECTIVE_PROMPT_TOKEN_LIMIT = 500 
        estimated_prompt_tokens = len(prompt_for_llm) / 3.5
        
        if estimated_prompt_tokens > EFFECTIVE_PROMPT_TOKEN_LIMIT:
            print(f"\n[API Ctx Mgmt] Est. prompt tokens ({int(estimated_prompt_tokens)}) > limit ({EFFECTIVE_PROMPT_TOKEN_LIMIT}). Clearing KV cache.")
            clear_ret = rkllm_lib.rkllm_clear_kv_cache(llm_handle, 1) 
            if clear_ret == 0:
                print("[API Ctx Mgmt] KV cache cleared.")
                conversation_history_bytes = b""
                prompt_for_llm = conversation_history_bytes + current_user_turn_bytes + ASSISTANT_TURN_START
                print(f"[API Ctx Mgmt] History reset. New prompt est: {int(len(prompt_for_llm)/3.5)} tokens.")
            else:
                print(f"[API Ctx Mgmt] Error clearing KV cache (code: {clear_ret}).")
        
        current_assistant_response_parts.clear()
        while not stream_token_queue.empty():
            try: stream_token_queue.get_nowait()
            except queue.Empty: break
        
        rkllm_input_obj = RKLLMInput(); rkllm_input_obj.input_type = RKLLM_INPUT_PROMPT; rkllm_input_obj.prompt_input = ctypes.c_char_p(prompt_for_llm)
        rkllm_infer_params_obj = RKLLMInferParam(); rkllm_infer_params_obj.mode = RKLLM_INFER_GENERATE; rkllm_infer_params_obj.keep_history = 1
        
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        model_name_to_return = rkllm_params_global.model_path.decode().split('/')[-1] if rkllm_params_global else "rkllm_model"
        created_ts = int(time.time())
        
        worker_status = WorkerStatus() 
        main_thread_run_ret_indicator = 0 

        if stream_requested:
            input_obj_ptr = ctypes.byref(rkllm_input_obj)
            infer_params_obj_ptr = ctypes.byref(rkllm_infer_params_obj)
            worker_thread = threading.Thread(target=rkllm_inference_worker, 
                                             args=(llm_handle, input_obj_ptr, infer_params_obj_ptr, worker_status))
            print(f"[API Handler {threading.get_ident()}] Starting worker thread for rkllm_run. Timestamp: {time.time()}")
            worker_thread.start()
            
            def generate_stream_response_sse():
                
                
                print(f"[Stream SSE Gen {threading.get_ident()}] Starting generator. Timestamp: {time.time()}")
                stream_had_error_signal = False
                streamed_content_count = 0
                try:
                    while True:
                        try:
                            token_or_signal = stream_token_queue.get(timeout=45.0) 
                        except queue.Empty:
                            print(f"[Stream SSE Gen {threading.get_ident()}] Timeout waiting for item from queue. Timestamp: {time.time()}")
                            if worker_thread.is_alive() or (not worker_status.finished):
                                 print(f"[Stream SSE Gen {threading.get_ident()}] Worker thread status: alive={worker_thread.is_alive()}, finished_flag={worker_status.finished}. Assuming stall.")
                            elif worker_status.run_ret != 0:
                                 print(f"[Stream SSE Gen {threading.get_ident()}] Worker thread finished with error ({worker_status.run_ret}).")
                            stream_had_error_signal = True 
                            break 

                        if token_or_signal is None: 
                            print(f"[Stream SSE Gen {threading.get_ident()}] Received None (error signal). Timestamp: {time.time()}")
                            stream_had_error_signal = True
                            break 
                        
                        if token_or_signal == "__LLM_RUN_FINISH__":
                            print(f"[Stream SSE Gen {threading.get_ident()}] Received FINISH signal. Timestamp: {time.time()}")
                            final_chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                            yield f"data: {json.dumps(final_chunk)}\n\n"
                            break
                        
                        streamed_content_count +=1
                        chunk = {"id": request_id, "object": "chat.completion.chunk", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "delta": {"content": token_or_signal}, "finish_reason": None}]}
                        yield f"data: {json.dumps(chunk)}\n\n"
                
                except Exception as e_stream:
                    print(f"[Stream SSE Gen {threading.get_ident()}] Error: {e_stream}. Timestamp: {time.time()}")
                    stream_had_error_signal = True
                finally:
                    print(f"[Stream SSE Gen {threading.get_ident()}] Sending [DONE]. Total content chunks: {streamed_content_count}. Timestamp: {time.time()}")
                    yield f"data: [DONE]\n\n"
                    
                    print(f"[Stream SSE Gen {threading.get_ident()}] Waiting for worker thread to complete. Timestamp: {time.time()}")
                    worker_thread.join(timeout=10.0) 
                    if worker_thread.is_alive():
                        print(f"[Stream SSE Gen {threading.get_ident()}] WARNING: Worker thread did not complete in time after stream.")
                        stream_had_error_signal = True 

                    if worker_status.finished and worker_status.run_ret == 0 and not stream_had_error_signal:
                        full_response_this_turn = "".join(current_assistant_response_parts) 
                        if full_response_this_turn: 
                            new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + full_response_this_turn.encode('utf-8') + ASSISTANT_TURN_END
                            conversation_history_bytes += new_segment
                            
                        else:
                            print(f"[API History Update after stream] No content in current_assistant_response_parts to update history.")
                    else:
                        print(f"[API History Update after stream] Skipped. Worker_finished: {worker_status.finished}, worker_run_ret: {worker_status.run_ret}, stream_had_error: {stream_had_error_signal}")
            
            return Response(stream_with_context(generate_stream_response_sse()), mimetype='text/event-stream')
        
        else: # One-shot response (synchronous rkllm_run)
            print(f"[API Handler {threading.get_ident()}] Calling rkllm_run (sync). Timestamp: {time.time()}")
            run_start_time = time.time()
            main_thread_run_ret_indicator = rkllm_lib.rkllm_run(llm_handle, ctypes.byref(rkllm_input_obj), ctypes.byref(rkllm_infer_params_obj), None)
            run_end_time = time.time()
            print(f"[API Handler {threading.get_ident()}] rkllm_run (sync) returned: {main_thread_run_ret_indicator}. Duration: {run_end_time - run_start_time:.4f}s. Qsize: {stream_token_queue.qsize()}. Timestamp: {time.time()}")

            if main_thread_run_ret_indicator != 0:
                print(f"[API Error] rkllm_run (sync) failed (code {main_thread_run_ret_indicator}).")
                return format_openai_error_response(f"RKLLM run failed (code: {main_thread_run_ret_indicator}).", "api_error"), 500

            full_assistant_response_str = "".join(current_assistant_response_parts)
            if main_thread_run_ret_indicator == 0: 
                new_segment = current_user_turn_bytes + ASSISTANT_TURN_START + full_assistant_response_str.encode('utf-8') + ASSISTANT_TURN_END
                conversation_history_bytes += new_segment

            prompt_tokens_est = len(prompt_for_llm) // 4 
            completion_tokens_est = len(full_assistant_response_str) // 4
            response_json = {"id": request_id, "object": "chat.completion", "created": created_ts, "model": model_name_to_return, "choices": [{"index": 0, "message": {"role": "assistant", "content": full_assistant_response_str}, "finish_reason": "stop" }], "usage": {"prompt_tokens": prompt_tokens_est, "completion_tokens": completion_tokens_est, "total_tokens": prompt_tokens_est + completion_tokens_est}}
            return jsonify(response_json)

# --- Main Entry Point ---
if __name__ == '__main__':
    if init_rkllm_model():
        print("Starting Flask server for RKLLM OpenAI-compliant API on http://0.0.0.0:1306/v1/chat/completions")
        app.run(host='0.0.0.0', port=1306, threaded=True, debug=False) 
    else:
        print("Failed to initialize RKLLM model. Server not starting.")

