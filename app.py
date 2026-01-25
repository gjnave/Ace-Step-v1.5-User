"""
ACE-Step v1.5 - HuggingFace Space Entry Point

This file serves as the entry point for HuggingFace Space deployment.
It initializes the service and launches the Gradio interface.
"""
import os
import sys

# Get current directory (app.py location)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Add nano-vllm to Python path (local package)
nano_vllm_path = os.path.join(current_dir, "acestep", "third_parts", "nano-vllm")
if os.path.exists(nano_vllm_path):
    sys.path.insert(0, nano_vllm_path)

# Disable Gradio analytics
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Clear proxy settings that may affect Gradio
for proxy_var in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY']:
    os.environ.pop(proxy_var, None)

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.dataset_handler import DatasetHandler
from acestep.gradio_ui import create_gradio_interface


def get_gpu_memory_gb():
    """
    Get GPU memory in GB. Returns 0 if no GPU is available.
    """
    try:
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            memory_gb = total_memory / (1024**3)
            return memory_gb
        else:
            return 0
    except Exception as e:
        print(f"Warning: Failed to detect GPU memory: {e}", file=sys.stderr)
        return 0


def main():
    """Main entry point for HuggingFace Space"""
    
    # HuggingFace Space persistent storage path
    persistent_storage_path = "/data"
    
    # Detect GPU memory for auto-configuration
    gpu_memory_gb = get_gpu_memory_gb()
    auto_offload = gpu_memory_gb > 0 and gpu_memory_gb < 16
    
    if auto_offload:
        print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB (< 16GB)")
        print("Auto-enabling CPU offload to reduce GPU memory usage")
    elif gpu_memory_gb > 0:
        print(f"Detected GPU memory: {gpu_memory_gb:.2f} GB (>= 16GB)")
        print("CPU offload disabled by default")
    else:
        print("No GPU detected, running on CPU")
    
    # Create handler instances
    print("Creating handlers...")
    dit_handler = AceStepHandler(persistent_storage_path=persistent_storage_path)
    llm_handler = LLMHandler(persistent_storage_path=persistent_storage_path)
    dataset_handler = DatasetHandler()
    
    # Service mode configuration from environment variables
    config_path = os.environ.get(
        "SERVICE_MODE_DIT_MODEL",
        "acestep-v15-turbo-fix-inst-shift-dynamic"
    )
    lm_model_path = os.environ.get(
        "SERVICE_MODE_LM_MODEL",
        "acestep-5Hz-lm-1.7B-v4-fix"
    )
    backend = os.environ.get("SERVICE_MODE_BACKEND", "vllm")
    device = "auto"
    
    print(f"Service mode configuration:")
    print(f"  DiT model: {config_path}")
    print(f"  LM model: {lm_model_path}")
    print(f"  Backend: {backend}")
    print(f"  Offload to CPU: {auto_offload}")
    
    # Determine flash attention availability
    use_flash_attention = dit_handler.is_flash_attention_available()
    print(f"  Flash Attention: {use_flash_attention}")
    
    # Initialize DiT model
    print(f"Initializing DiT model: {config_path}...")
    init_status, enable_generate = dit_handler.initialize_service(
        project_root=current_dir,
        config_path=config_path,
        device=device,
        use_flash_attention=use_flash_attention,
        compile_model=False,
        offload_to_cpu=auto_offload,
        offload_dit_to_cpu=False
    )
    
    if not enable_generate:
        print(f"Warning: DiT model initialization issue: {init_status}", file=sys.stderr)
    else:
        print("DiT model initialized successfully")
    
    # Initialize LM model
    checkpoint_dir = dit_handler._get_checkpoint_dir()
    print(f"Initializing 5Hz LM: {lm_model_path}...")
    lm_status, lm_success = llm_handler.initialize(
        checkpoint_dir=checkpoint_dir,
        lm_model_path=lm_model_path,
        backend=backend,
        device=device,
        offload_to_cpu=auto_offload,
        dtype=dit_handler.dtype
    )
    
    if lm_success:
        print("5Hz LM initialized successfully")
        init_status += f"\n{lm_status}"
    else:
        print(f"Warning: 5Hz LM initialization failed: {lm_status}", file=sys.stderr)
        init_status += f"\n{lm_status}"
    
    # Prepare initialization parameters for UI
    init_params = {
        'pre_initialized': True,
        'service_mode': True,
        'checkpoint': None,
        'config_path': config_path,
        'device': device,
        'init_llm': True,
        'lm_model_path': lm_model_path,
        'backend': backend,
        'use_flash_attention': use_flash_attention,
        'offload_to_cpu': auto_offload,
        'offload_dit_to_cpu': False,
        'init_status': init_status,
        'enable_generate': enable_generate,
        'dit_handler': dit_handler,
        'llm_handler': llm_handler,
        'language': 'en',
        'persistent_storage_path': persistent_storage_path,
    }
    
    print("Service initialization completed!")
    
    # Create Gradio interface with pre-initialized handlers
    print("Creating Gradio interface...")
    demo = create_gradio_interface(
        dit_handler, 
        llm_handler, 
        dataset_handler, 
        init_params=init_params, 
        language='en'
    )
    
    # Enable queue for multi-user support
    print("Enabling queue for multi-user support...")
    demo.queue(max_size=20)
    
    # Launch
    print("Launching server on 0.0.0.0:7860...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
