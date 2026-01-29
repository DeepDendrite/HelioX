#!/usr/bin/env python3
"""
Test script for enhanced export functionality

This script demonstrates the enhanced export/import functionality
without requiring NEURON to be installed.
"""

import os
import sys
import json
import tempfile
import shutil

# Add current directory to path for local imports
sys.path.insert(0, '$HOME/heliox/python_lib')

import heliox_export


def test_metadata_serialization():
    """Test metadata serialization functions"""
    print("Testing metadata serialization...")
    
    # Test metadata structure creation
    metadata = {
        "version": "1.0",
        "export_timestamp": heliox_export.datetime.datetime.now().isoformat(),
        "wrappers": {
            "monitors": [
                {
                    "id": "test-monitor-1", 
                    "mech_name": "global",
                    "var_name": "v",
                    "node_or_mech_idx": 0
                }
            ],
            "obj_wrappers": [
                {
                    "id": "test-obj-1",
                    "mech_name": "IClamp", 
                    "node_or_mech_idx": 0,
                    "allowed_vars": ["amp", "dur", "delay"],
                    "is_segment": False
                }
            ],
            "vecplay_wrappers": [
                {
                    "id": "test-vecplay-1",
                    "mech_name": "IClamp",
                    "var_name": "amp", 
                    "instance_id": 0
                }
            ]
        }
    }
    
    # Test validation
    if heliox_export.validate_metadata(metadata):
        print("✓ Metadata validation passed")
    else:
        print("✗ Metadata validation failed")
        return False
    
    # Test configuration
    config = {
        "device": "gpu",
        "permute_type": 3,
        "dt": 0.05,
        "v_init": -62.5
    }
    
    if heliox_export.validate_config(config):
        print("✓ Configuration validation passed")
    else:
        print("✗ Configuration validation failed")
        return False
    
    return True


def test_file_operations():
    """Test file save/load operations"""
    print("\nTesting file operations...")
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test data
        metadata = {
            "version": "1.0",
            "export_timestamp": heliox_export.datetime.datetime.now().isoformat(),
            "wrappers": {
                "monitors": [],
                "obj_wrappers": [],
                "vecplay_wrappers": []
            }
        }
        
        config = {
            "device": "gpu",
            "permute_type": 3,
            "dt": 0.05,
            "v_init": -62.5
        }
        
        # Test metadata save/load
        metadata_path = os.path.join(temp_dir, "heliox_metadata.json")
        heliox_export.save_metadata_to_file(metadata, metadata_path)
        
        if os.path.exists(metadata_path):
            print("✓ Metadata file created successfully")
        else:
            print("✗ Metadata file creation failed")
            return False
        
        loaded_metadata = heliox_export.load_metadata_from_file(metadata_path)
        if loaded_metadata == metadata:
            print("✓ Metadata save/load roundtrip successful")
        else:
            print("✗ Metadata save/load roundtrip failed")
            return False
        
        # Test config save/load
        config_path = os.path.join(temp_dir, "heliox_config.json")
        heliox_export.save_config_to_file(config, config_path)
        
        if os.path.exists(config_path):
            print("✓ Configuration file created successfully")
        else:
            print("✗ Configuration file creation failed")
            return False
        
        loaded_config = heliox_export.load_config_from_file(config_path)
        if loaded_config == config:
            print("✓ Configuration save/load roundtrip successful")
        else:
            print("✗ Configuration save/load roundtrip failed")
            return False
    
    return True


def test_unique_id_generation():
    """Test unique ID generation"""
    print("\nTesting unique ID generation...")
    
    # Generate multiple IDs and check they're unique
    ids = set()
    for i in range(100):
        id_val = heliox_export.generate_unique_id()
        if id_val in ids:
            print("✗ Duplicate ID generated")
            return False
        ids.add(id_val)
    
    print("✓ Unique ID generation test passed")
    return True


def test_export_import_workflow():
    """Test the complete export/import workflow (without heliox)"""
    print("\nTesting export/import workflow...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Create sample export files
        metadata = {
            "version": "1.0", 
            "export_timestamp": heliox_export.datetime.datetime.now().isoformat(),
            "wrappers": {
                "monitors": [
                    {
                        "id": heliox_export.generate_unique_id(),
                        "mech_name": "global",
                        "var_name": "v", 
                        "node_or_mech_idx": 0
                    }
                ],
                "obj_wrappers": [
                    {
                        "id": heliox_export.generate_unique_id(),
                        "mech_name": "IClamp",
                        "node_or_mech_idx": 0,
                        "allowed_vars": ["amp", "dur", "delay"],
                        "is_segment": False
                    }
                ],
                "vecplay_wrappers": [
                    {
                        "id": heliox_export.generate_unique_id(),
                        "mech_name": "IClamp",
                        "var_name": "amp",
                        "instance_id": 0
                    }
                ]
            }
        }
        
        config = {
            "device": "gpu",
            "permute_type": 3,
            "dt": 0.05,
            "v_init": -62.5
        }
        
        # Save files
        metadata_path = os.path.join(temp_dir, "heliox_metadata.json")
        config_path = os.path.join(temp_dir, "heliox_config.json")
        
        heliox_export.save_metadata_to_file(metadata, metadata_path)
        heliox_export.save_config_to_file(config, config_path)
        
        print("✓ Export files created successfully")
        
        # Test load (this would normally create wrappers, but we can't without heliox)
        loaded_metadata = heliox_export.load_metadata_from_file(metadata_path)
        loaded_config = heliox_export.load_config_from_file(config_path)
        
        if (heliox_export.validate_metadata(loaded_metadata) and 
            heliox_export.validate_config(loaded_config)):
            print("✓ Import validation successful")
        else:
            print("✗ Import validation failed")
            return False
    
    return True


def main():
    """Run all tests"""
    print("Enhanced Export Functionality Test Suite")
    print("=" * 50)
    
    tests = [
        test_metadata_serialization,
        test_file_operations, 
        test_unique_id_generation,
        test_export_import_workflow
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! Enhanced export functionality is working correctly.")
        return 0
    else:
        print("✗ Some tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())