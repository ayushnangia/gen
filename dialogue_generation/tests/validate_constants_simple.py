#!/usr/bin/env python3
"""
Simple validation test for constants extraction
Compares extracted constants with source code using AST parsing
"""

import sys
import os
import ast

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def extract_constant_from_file(file_path: str, constant_name: str):
    """Extract constant value from Python file using AST parsing"""
    with open(file_path, 'r', encoding='utf-8') as f:
        source_code = f.read()

    tree = ast.parse(source_code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == constant_name:
                    return ast.literal_eval(node.value)
    return None

def test_constants_extraction_simple():
    """Test constants extraction by comparing with original source files"""

    print("🔍 Validating constants extraction (simple version)...")

    # Import from new module
    from dialogue_generation.config.constants import (
        USER_EMOTION_LIST, ASSISTANT_EMOTION_LIST, SCENARIO_CATEGORIES,
        PREDEFINED_REGIONS, TRAVEL_TIME_SLOTS, RESOLUTION_STATUSES,
        SERVICE_WEIGHTS, CORE_SERVICES, LOGICAL_COMBINATIONS
    )

    # File paths
    serial_file = 'serial_gen.py'
    parallel_file = 'parallel_gen.py'

    # Constants to validate
    constants_to_check = [
        ('USER_EMOTION_LIST', USER_EMOTION_LIST),
        ('ASSISTANT_EMOTION_LIST', ASSISTANT_EMOTION_LIST),
        ('SCENARIO_CATEGORIES', SCENARIO_CATEGORIES),
        ('PREDEFINED_REGIONS', PREDEFINED_REGIONS),
    ]

    all_passed = True

    for constant_name, extracted_value in constants_to_check:
        print(f"  🔍 Checking {constant_name}...")

        # Extract from serial_gen.py
        try:
            serial_value = extract_constant_from_file(serial_file, constant_name)
            if serial_value is not None:
                if extracted_value == serial_value:
                    print(f"    ✅ {constant_name} matches serial_gen.py")
                else:
                    print(f"    ❌ {constant_name} MISMATCH with serial_gen.py")
                    all_passed = False
            else:
                print(f"    ⚠️ {constant_name} not found in serial_gen.py")
        except Exception as e:
            print(f"    ⚠️ Error extracting {constant_name} from serial_gen.py: {e}")

        # Extract from parallel_gen.py
        try:
            parallel_value = extract_constant_from_file(parallel_file, constant_name)
            if parallel_value is not None:
                if extracted_value == parallel_value:
                    print(f"    ✅ {constant_name} matches parallel_gen.py")
                else:
                    print(f"    ❌ {constant_name} MISMATCH with parallel_gen.py")
                    all_passed = False
            else:
                print(f"    ⚠️ {constant_name} not found in parallel_gen.py")
        except Exception as e:
            print(f"    ⚠️ Error extracting {constant_name} from parallel_gen.py: {e}")

    # Validate extracted constant properties
    print("  🔍 Validating extracted constant properties...")

    # Validate list lengths
    try:
        assert len(USER_EMOTION_LIST) == 20, f"USER_EMOTION_LIST should have 20 items, got {len(USER_EMOTION_LIST)}"
        print(f"    ✅ USER_EMOTION_LIST has correct length: {len(USER_EMOTION_LIST)}")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    try:
        assert len(ASSISTANT_EMOTION_LIST) == 20, f"ASSISTANT_EMOTION_LIST should have 20 items, got {len(ASSISTANT_EMOTION_LIST)}"
        print(f"    ✅ ASSISTANT_EMOTION_LIST has correct length: {len(ASSISTANT_EMOTION_LIST)}")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    try:
        assert len(PREDEFINED_REGIONS) == 55, f"PREDEFINED_REGIONS should have 55 items, got {len(PREDEFINED_REGIONS)}"
        print(f"    ✅ PREDEFINED_REGIONS has correct length: {len(PREDEFINED_REGIONS)}")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    try:
        assert len(TRAVEL_TIME_SLOTS) == 6, f"TRAVEL_TIME_SLOTS should have 6 items, got {len(TRAVEL_TIME_SLOTS)}"
        print(f"    ✅ TRAVEL_TIME_SLOTS has correct length: {len(TRAVEL_TIME_SLOTS)}")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    # Validate dict structures
    try:
        expected_scenario_categories = {'general', 'restaurant', 'hotel', 'train', 'attraction', 'taxi', 'hospital', 'bus', 'flight'}
        assert set(SCENARIO_CATEGORIES.keys()) == expected_scenario_categories, f"SCENARIO_CATEGORIES keys mismatch"
        print(f"    ✅ SCENARIO_CATEGORIES has correct structure: {len(SCENARIO_CATEGORIES)} categories")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    # Validate resolution status probabilities sum to 1.0
    try:
        resolution_sum = sum(RESOLUTION_STATUSES.values())
        assert abs(resolution_sum - 1.0) < 0.001, f"Resolution statuses should sum to 1.0, got {resolution_sum}"
        print(f"    ✅ RESOLUTION_STATUSES probabilities sum correctly: {resolution_sum}")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    # Validate service weights sum to 1.0
    try:
        service_weights_sum = sum(SERVICE_WEIGHTS.values())
        assert abs(service_weights_sum - 1.0) < 0.001, f"Service weights should sum to 1.0, got {service_weights_sum}"
        print(f"    ✅ SERVICE_WEIGHTS sum correctly: {service_weights_sum}")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    # Validate logical combinations structure
    try:
        assert 'double' in LOGICAL_COMBINATIONS, "Missing 'double' in LOGICAL_COMBINATIONS"
        assert 'triple' in LOGICAL_COMBINATIONS, "Missing 'triple' in LOGICAL_COMBINATIONS"
        assert 'quadruple' in LOGICAL_COMBINATIONS, "Missing 'quadruple' in LOGICAL_COMBINATIONS"
        print(f"    ✅ LOGICAL_COMBINATIONS has correct structure")
    except AssertionError as e:
        print(f"    ❌ {e}")
        all_passed = False

    if all_passed:
        print("\n🎉 ALL CONSTANTS VALIDATION PASSED!")
        print("✅ Constants extracted successfully - safe to proceed to next phase")
        return True
    else:
        print("\n❌ CONSTANTS VALIDATION FAILED!")
        print("🚨 DO NOT PROCEED - Fix constants extraction first")
        return False

def show_constants_summary():
    """Show summary of extracted constants"""
    from dialogue_generation.config.constants import (
        USER_EMOTION_LIST, ASSISTANT_EMOTION_LIST, SCENARIO_CATEGORIES,
        PREDEFINED_REGIONS, TRAVEL_TIME_SLOTS, RESOLUTION_STATUSES,
        SERVICE_WEIGHTS, CORE_SERVICES, LOGICAL_COMBINATIONS
    )

    print("\n📊 CONSTANTS SUMMARY:")
    print("=" * 50)
    print(f"User emotions: {len(USER_EMOTION_LIST)} items")
    print(f"  Sample: {USER_EMOTION_LIST[:3]}...")
    print(f"Assistant emotions: {len(ASSISTANT_EMOTION_LIST)} items")
    print(f"  Sample: {ASSISTANT_EMOTION_LIST[:3]}...")
    print(f"Scenario categories: {len(SCENARIO_CATEGORIES)} service types")
    print(f"  Categories: {list(SCENARIO_CATEGORIES.keys())}")
    print(f"Predefined regions: {len(PREDEFINED_REGIONS)} cities")
    print(f"  Sample: {PREDEFINED_REGIONS[:5]}...")
    print(f"Travel time slots: {len(TRAVEL_TIME_SLOTS)} periods")
    print(f"  Slots: {[slot[2] for slot in TRAVEL_TIME_SLOTS]}")
    print(f"Resolution statuses: {len(RESOLUTION_STATUSES)} types")
    print(f"  Statuses: {list(RESOLUTION_STATUSES.keys())}")
    print(f"Core services: {len(CORE_SERVICES)} services")
    print(f"  Services: {CORE_SERVICES}")
    print(f"Service combinations:")
    for combo_type, combos in LOGICAL_COMBINATIONS.items():
        print(f"  - {combo_type}: {len(combos)} combinations")
    print("=" * 50)

if __name__ == "__main__":
    try:
        success = test_constants_extraction_simple()
        if success:
            show_constants_summary()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)