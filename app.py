"""
ACE-Step v1.5 - HuggingFace Space Entry Point

This file serves as the entry point for HuggingFace Space deployment.
It imports and uses the existing v1.5 Gradio implementation without modification.
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

from acestep.acestep_v15_pipeline import create_demo


def main():
    """Main entry point for HuggingFace Space"""

    # HuggingFace Space initialization parameters
    init_params = {
        'pre_initialized': False,  # Lazy initialization
        'service_mode': True,      # Service mode
        'language': 'en',
        'persistent_storage_path': '/data',  # HuggingFace Space persistent storage
    }

    # Create demo using existing v1.5 implementation
    demo = create_demo(init_params=init_params, language='en')

    # Enable queue for multi-user support
    demo.queue(max_size=20)

    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
