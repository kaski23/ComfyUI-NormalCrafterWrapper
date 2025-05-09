# ComfyUI/custom_nodes/ComfyUI-NormalCrafter/__init__.py

try:
    from .normal_crafter_nodes import NormalCrafterNode

    NODE_CLASS_MAPPINGS = {
        "NormalCrafterNode": NormalCrafterNode,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "NormalCrafterNode": "NormalCrafter (Process Video)",
    }

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    print("✅ ComfyUI-NormalCrafter: Custom nodes loaded successfully.")

except ImportError as e:
    print(f"❌ ComfyUI-NormalCrafter: Failed to import nodes: {e}")
    print("   Please ensure all dependencies are installed (e.g., diffusers, transformers, huggingface_hub) and the files are correctly placed.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}