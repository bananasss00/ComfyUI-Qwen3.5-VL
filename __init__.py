import os
import sys
import subprocess
import logging

def setup_vendored_dependencies():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vendor_dir = os.path.join(current_dir, "vendor")
    check_file = os.path.join(vendor_dir, "transformers", "__init__.py")
    
    if not os.path.exists(check_file):
        logging.info("[Qwen3.5-VL] Vendored dependencies not found. Starting installation...")
        os.makedirs(vendor_dir, exist_ok=True)
        
        command =[
            sys.executable, "-m", "pip", "install",
            "transformers==5.2.0",
            "huggingface_hub",
            "--target", vendor_dir,
            "--upgrade",
            "--no-deps",
            "--quiet"
        ]
        
        try:
            subprocess.run(command, check=True)
            logging.info("[Qwen3.5-VL] Dependencies successfully installed to the isolated environment.")
        except subprocess.CalledProcessError as e:
            logging.error(f"[Qwen3.5-VL] Dependency installation failed: {e}")
            logging.error("Please install them manually using: pip install transformers==5.2.0 huggingface_hub tokenizers --target vendor")

setup_vendored_dependencies()


from . import nodes

NODE_CLASS_MAPPINGS = {
    **nodes.NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **nodes.NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]