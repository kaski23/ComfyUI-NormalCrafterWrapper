# ComfyUI/custom_nodes/ComfyUI-NormalCrafter/__init__.py
# Or ComfyUI/custom_nodes/ComfyUI-NormalCrafterWrapper/__init__.py

try:
    # Import BOTH classes from your nodes file
    from .normal_crafter_nodes import NormalCrafterNode, DetailTransfer # <--- IMPORT DetailTransfer HERE

    NODE_CLASS_MAPPINGS = {
        "NormalCrafterNode": NormalCrafterNode,
        "DetailTransfer": DetailTransfer  # Now DetailTransfer is defined
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "NormalCrafterNode": "NormalCrafter (Process Video)", # Or your preferred name
        "DetailTransfer": "Detail Transfer"
    }

    __all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
    print("✅ ComfyUI-NormalCrafter: Custom nodes loaded successfully.")

except ImportError as e:
    print(f"❌ ComfyUI-NormalCrafter: Failed to import nodes: {e}")
    print("   Please ensure all dependencies are installed (e.g., diffusers, transformers, huggingface_hub) and the files are correctly placed.")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
except NameError as e: # Catch NameError specifically if DetailTransfer wasn't found during import
    print(f"❌ ComfyUI-NormalCrafter: Failed to define nodes, likely an issue importing a class: {e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
