#!/usr/bin/env python3
"""
Debug script to check configuration loading
"""

import os
from dotenv import load_dotenv
from config.config import FinSightConfig

def debug_config_loading():
    print("=== Configuration Debug ===")
    
    # Check environment before load_dotenv
    print("\n1. Environment BEFORE load_dotenv():")
    print(f"   SEC_API_KEY: {'Set' if os.getenv('SEC_API_KEY') else 'Not Set'}")
    
    # Load environment variables
    load_dotenv()
    print("\n2. Environment AFTER load_dotenv():")
    sec_key_raw = os.getenv('SEC_API_KEY')
    print(f"   SEC_API_KEY: {'Set' if sec_key_raw else 'Not Set'}")
    if sec_key_raw:
        print(f"   Key preview: {sec_key_raw[:4]}...{sec_key_raw[-4:]} (length: {len(sec_key_raw)})")
        print(f"   Key repr: {repr(sec_key_raw[:20])}")  # Shows whitespace/special chars
    
    # Check .env file existence and content
    print("\n3. .env file check:")
    env_path = ".env"
    if os.path.exists(env_path):
        print(f"   ✓ .env file exists at: {os.path.abspath(env_path)}")
        try:
            with open(env_path, 'r') as f:
                content = f.read()
                lines = content.split('\n')
                sec_lines = [line for line in lines if 'SEC_API_KEY' in line]
                if sec_lines:
                    print(f"   SEC_API_KEY lines found: {len(sec_lines)}")
                    for i, line in enumerate(sec_lines):
                        print(f"     Line {i+1}: {repr(line)}")
                else:
                    print("   ✗ No SEC_API_KEY found in .env file")
        except Exception as e:
            print(f"   ✗ Error reading .env file: {e}")
    else:
        print(f"   ✗ .env file not found at: {os.path.abspath(env_path)}")
    
    # Test FinSightConfig
    print("\n4. FinSightConfig test:")
    try:
        config = FinSightConfig()
        print(f"   config.sec_api_key: {'Set' if config.sec_api_key else 'Not Set'}")
        if config.sec_api_key:
            print(f"   Key preview: {config.sec_api_key[:4]}...{config.sec_api_key[-4:]} (length: {len(config.sec_api_key)})")
        
        # Test with explicit key
        print("\n5. Test with explicit key:")
        explicit_config = FinSightConfig(sec_api_key="test_key_123")
        print(f"   explicit config.sec_api_key: {'Set' if explicit_config.sec_api_key else 'Not Set'}")
        if explicit_config.sec_api_key:
            print(f"   Explicit key: {explicit_config.sec_api_key}")
            
    except Exception as e:
        print(f"   ✗ Error creating FinSightConfig: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_config_loading() 