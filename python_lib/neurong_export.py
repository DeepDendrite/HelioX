"""
NeurongExport - Enhanced Export/Import Functionality
================================================

This module provides enhanced export/import functionality for the NEURON-compatible
simulator's wrapper system. It allows saving wrapper metadata alongside NEURON's
.dat files, enabling model loading without NEURON after export.

Key Features:
-------------
1. Serialize wrapper metadata to JSON format
2. Export both NEURON model and wrapper information
3. Load models and recreate wrappers from metadata alone
4. Backward compatibility with existing export behavior

Usage:
------
```python
from neurong_wrapper import NeurongManager

# Create and configure wrappers
neurong_manager = NeurongManager()
# ... create wrappers ...

# Enhanced export with metadata
neurong_manager.enhanced_export_model("path/to/export", dt=0.05, v_init=-62.5)

# Load from exported metadata (no NEURON needed)
neurong_manager.load_from_export("path/to/export")
```

File Structure:
---------------
export_directory/
├── neurong_metadata.json    # Wrapper information
├── neurong_config.json      # Configuration (device, dt, etc.)
└── *.dat files             # NEURON exported data
"""

import os
import json
import datetime
import uuid
from typing import Dict, List, Any, Optional


def generate_unique_id() -> str:
    """Generate a unique ID for wrapper identification"""
    return str(uuid.uuid4())


def serialize_monitor_wrapper(wrapper) -> Dict[str, Any]:
    """
    Serialize a MonitorWrapper to dictionary format
    
    Parameters:
    -----------
    wrapper : MonitorWrapper
        The monitor wrapper to serialize
        
    Returns:
    --------
    dict
        Serialized wrapper data
    """
    if not wrapper.initialized:
        raise ValueError("Cannot serialize uninitialized MonitorWrapper")
    
    return {
        "id": generate_unique_id(),
        "mech_name": wrapper._mech_name,
        "var_name": wrapper.var_name,
        "node_or_mech_idx": wrapper._node_or_mech_idx,
        "array_index": getattr(wrapper, "array_index", 0),
    }


def serialize_window_recorder_wrapper(wrapper) -> Dict[str, Any]:
    """Serialize a WindowRecorderWrapper to dictionary format."""
    return serialize_monitor_wrapper(wrapper)


def serialize_obj_wrapper(wrapper) -> Dict[str, Any]:
    """
    Serialize an ObjWrapper to dictionary format
    
    Parameters:
    -----------
    wrapper : ObjWrapper
        The object wrapper to serialize
        
    Returns:
    --------
    dict
        Serialized wrapper data
    """
    if not wrapper.initialized:
        raise ValueError("Cannot serialize uninitialized ObjWrapper")
    
    # Access private attributes using name mangling
    mech_name = wrapper._ObjWrapper__mech_name
    allowed_vars = list(wrapper._ObjWrapper__allowed_vars)
    is_segment = wrapper._ObjWrapper__is_segment
    row = wrapper._ObjWrapper__row
    
    return {
        "id": generate_unique_id(),
        "mech_name": mech_name,
        "node_or_mech_idx": row,
        "allowed_vars": allowed_vars,
        "is_segment": is_segment
    }


def serialize_vecplay_wrapper(wrapper) -> Dict[str, Any]:
    """
    Serialize a VecPlayWrapper to dictionary format
    
    Parameters:
    -----------
    wrapper : VecPlayWrapper
        The VecPlay wrapper to serialize
        
    Returns:
    --------
    dict
        Serialized wrapper data
    """
    if not wrapper.initialized:
        raise ValueError("Cannot serialize uninitialized VecPlayWrapper")
    
    return {
        "id": generate_unique_id(),
        "mech_name": wrapper._mech_name,
        "var_name": wrapper.var_name,
        "instance_id": wrapper._instance_id
    }


def serialize_all_wrappers(neurong_manager) -> Dict[str, Any]:
    """
    Serialize all wrappers from NeurongManager
    
    Parameters:
    -----------
    neurong_manager : NeurongManager
        The manager containing all wrappers
        
    Returns:
    --------
    dict
        Complete wrapper metadata
    """
    metadata = {
        "version": "1.0",
        "export_timestamp": datetime.datetime.now().isoformat(),
        "wrappers": {
            "monitors": [],
            "window_recorders": [],
            "obj_wrappers": [],
            "vecplay_wrappers": []
        }
    }
    
    # Serialize monitor wrappers
    # Note: We can't directly access all created wrappers, so this would need
    # to be called during the export process when wrappers are available
    # For now, we'll return the structure and populate it in the manager
    
    return metadata


def save_metadata_to_file(metadata: Dict[str, Any], filepath: str):
    """
    Save metadata dictionary to JSON file
    
    Parameters:
    -----------
    metadata : dict
        The metadata to save
    filepath : str
        Path to save the JSON file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)


def load_metadata_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
        
    Returns:
    --------
    dict
        Loaded metadata
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metadata file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def save_config_to_file(config: Dict[str, Any], filepath: str):
    """
    Save configuration to JSON file
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary
    filepath : str
        Path to save the configuration file
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Parameters:
    -----------
    filepath : str
        Path to the configuration file
        
    Returns:
    --------
    dict
        Loaded configuration
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        return json.load(f)


def create_standalone_monitor_wrapper(metadata: Dict[str, Any], client):
    """
    Create a MonitorWrapper from metadata without NEURON object
    
    Parameters:
    -----------
    metadata : dict
        Serialized monitor wrapper data
    client : neurong.Sim
        Neurong client instance
        
    Returns:
    --------
    MonitorWrapper
        Reconstructed wrapper (standalone)
    """
    # Import here to avoid circular imports
    from neurong_wrapper import MonitorWrapper
    
    # Create a wrapper with None object (standalone mode)
    wrapper = MonitorWrapper(None, metadata["var_name"], client, metadata.get("array_index", 0))
    
    # Set the internal state directly
    wrapper._mech_name = metadata["mech_name"]
    wrapper.var_name = metadata["var_name"]
    wrapper._node_or_mech_idx = metadata["node_or_mech_idx"]
    wrapper.array_index = metadata.get("array_index", 0)
    wrapper.monitor_id = -1  # Will be resolved when accessing data
    wrapper.initialized = True
    wrapper.obj = None  # Mark as standalone
    
    return wrapper


def create_standalone_window_recorder_wrapper(metadata: Dict[str, Any], client):
    """Create a WindowRecorderWrapper from metadata without NEURON object."""
    from neurong_wrapper import WindowRecorderWrapper

    wrapper = WindowRecorderWrapper(None, metadata["var_name"], client, metadata.get("array_index", 0))
    wrapper.obj = None
    wrapper._mech_name = metadata["mech_name"]
    wrapper.var_name = metadata["var_name"]
    wrapper._node_or_mech_idx = metadata["node_or_mech_idx"]
    wrapper.array_index = metadata.get("array_index", 0)
    wrapper.window_observer_id = -1
    wrapper.initialized = True
    return wrapper


def create_standalone_obj_wrapper(metadata: Dict[str, Any], client):
    """
    Create an ObjWrapper from metadata without NEURON object
    
    Parameters:
    -----------
    metadata : dict
        Serialized object wrapper data
    client : neurong.Sim
        Neurong client instance
        
    Returns:
    --------
    ObjWrapper
        Reconstructed wrapper (standalone)
    """
    # Import here to avoid circular imports
    from neurong_wrapper import ObjWrapper
    
    # Create a wrapper with None object (standalone mode)
    wrapper = ObjWrapper(None, client)
    
    # Set the internal state directly using name mangling
    wrapper._ObjWrapper__obj = None  # Mark as standalone
    wrapper._ObjWrapper__mech_name = metadata["mech_name"]
    wrapper._ObjWrapper__is_segment = metadata["is_segment"]
    wrapper._ObjWrapper__allowed_vars = set(metadata["allowed_vars"])
    wrapper._ObjWrapper__row = metadata["node_or_mech_idx"]
    wrapper.initialized = True
    
    # Recreate properties for all allowed variables
    for var_name in metadata["allowed_vars"]:
        wrapper._create_property(var_name)
    
    return wrapper


def create_standalone_vecplay_wrapper(metadata: Dict[str, Any], client):
    """
    Create a VecPlayWrapper from metadata without NEURON object
    
    Parameters:
    -----------
    metadata : dict
        Serialized VecPlay wrapper data
    client : neurong.Sim
        Neurong client instance
        
    Returns:
    --------
    VecPlayWrapper
        Reconstructed wrapper (standalone)
    """
    # Import here to avoid circular imports
    from neurong_wrapper import VecPlayWrapper
    
    # Create a wrapper with None object (standalone mode)
    wrapper = VecPlayWrapper(None, metadata["var_name"], client)
    
    # Set the internal state directly
    wrapper.obj = None  # Mark as standalone
    wrapper.var_name = metadata["var_name"]
    wrapper._mech_name = metadata["mech_name"]
    wrapper._instance_id = metadata["instance_id"]
    wrapper.initialized = True
    
    return wrapper


def validate_metadata(metadata: Dict[str, Any]) -> bool:
    """
    Validate metadata structure
    
    Parameters:
    -----------
    metadata : dict
        Metadata to validate
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    required_keys = ["version", "export_timestamp", "wrappers"]
    if not all(key in metadata for key in required_keys):
        return False
    
    wrapper_keys = ["monitors", "obj_wrappers", "vecplay_wrappers"]
    if not all(key in metadata["wrappers"] for key in wrapper_keys):
        return False
    
    return True


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure
    
    Parameters:
    -----------
    config : dict
        Configuration to validate
        
    Returns:
    --------
    bool
        True if valid, False otherwise
    """
    required_keys = ["device", "permute_type", "dt", "v_init"]
    return all(key in config for key in required_keys)
