#!/usr/bin/env python3
"""
Validation test for constants extraction
Ensures extracted constants match originals exactly
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_constants_extraction():
    """Ensure extracted constants match originals exactly"""

    print("🔍 Validating constants extraction...")

    # Import from new module
    from dialogue_generation.config.constants import (
        USER_EMOTION_LIST, ASSISTANT_EMOTION_LIST, SCENARIO_CATEGORIES,
        PREDEFINED_REGIONS, TRAVEL_TIME_SLOTS, RESOLUTION_STATUSES,
        SERVICE_WEIGHTS, CORE_SERVICES, LOGICAL_COMBINATIONS
    )

    # Import from original files to compare
    from serial_gen import DialogueGenerator as SerialGen
    from parallel_gen import DialogueGenerator as ParallelGen

    # Create instances to access constants
    serial_config = {'total_generations': 1}
    parallel_config = {'total_generations': 1}

    print("  📋 Creating generator instances for comparison...")
    serial_gen = SerialGen(serial_config)
    parallel_gen = ParallelGen(parallel_config)

    # Validate exact matches
    validations = [
        (USER_EMOTION_LIST, serial_gen.USER_EMOTION_LIST, "USER_EMOTION_LIST (serial)"),
        (USER_EMOTION_LIST, parallel_gen.USER_EMOTION_LIST, "USER_EMOTION_LIST (parallel)"),
        (ASSISTANT_EMOTION_LIST, serial_gen.ASSISTANT_EMOTION_LIST, "ASSISTANT_EMOTION_LIST (serial)"),
        (ASSISTANT_EMOTION_LIST, parallel_gen.ASSISTANT_EMOTION_LIST, "ASSISTANT_EMOTION_LIST (parallel)"),
        (SCENARIO_CATEGORIES, serial_gen.SCENARIO_CATEGORIES, "SCENARIO_CATEGORIES (serial)"),
        (SCENARIO_CATEGORIES, parallel_gen.SCENARIO_CATEGORIES, "SCENARIO_CATEGORIES (parallel)"),
        (PREDEFINED_REGIONS, serial_gen.PREDEFINED_REGIONS, "PREDEFINED_REGIONS (serial)"),
        (PREDEFINED_REGIONS, parallel_gen.PREDEFINED_REGIONS, "PREDEFINED_REGIONS (parallel)"),
        (TRAVEL_TIME_SLOTS, serial_gen.travel_time_slots, "TRAVEL_TIME_SLOTS (serial)"),
        (TRAVEL_TIME_SLOTS, parallel_gen.travel_time_slots, "TRAVEL_TIME_SLOTS (parallel)"),
        (RESOLUTION_STATUSES, serial_gen.RESOLUTION_STATUSES, "RESOLUTION_STATUSES (serial)"),
        (RESOLUTION_STATUSES, parallel_gen.RESOLUTION_STATUSES, "RESOLUTION_STATUSES (parallel)"),
    ]

    all_passed = True
    for new_val, original_val, name in validations:
        try:
            assert new_val == original_val, f"{name} mismatch!"
            print(f"  ✅ {name} matches exactly")
        except AssertionError as e:
            print(f"  ❌ {name} MISMATCH: {e}")
            all_passed = False

            # Show detailed comparison for debugging
            if isinstance(new_val, list) and isinstance(original_val, list):
                print(f"    New length: {len(new_val)}, Original length: {len(original_val)}")
                if len(new_val) != len(original_val):
                    print(f"    Length difference detected!")
                else:
                    for i, (new_item, orig_item) in enumerate(zip(new_val, original_val)):
                        if new_item != orig_item:
                            print(f"    Difference at index {i}: '{new_item}' vs '{orig_item}'")
                            break
            elif isinstance(new_val, dict) and isinstance(original_val, dict):
                print(f"    New keys: {set(new_val.keys())}")
                print(f"    Original keys: {set(original_val.keys())}")
                if set(new_val.keys()) != set(original_val.keys()):
                    print(f"    Key differences: {set(new_val.keys()) ^ set(original_val.keys())}")
                else:
                    for key in new_val.keys():
                        if new_val[key] != original_val[key]:
                            print(f"    Difference at key '{key}': {new_val[key]} vs {original_val[key]}")
                            break

    # Additional validations for extracted constants
    print("  🔍 Validating extracted constant properties...")

    # Validate list lengths are reasonable
    assert len(USER_EMOTION_LIST) == 20, f"USER_EMOTION_LIST should have 20 items, got {len(USER_EMOTION_LIST)}"
    assert len(ASSISTANT_EMOTION_LIST) == 20, f"ASSISTANT_EMOTION_LIST should have 20 items, got {len(ASSISTANT_EMOTION_LIST)}"
    assert len(PREDEFINED_REGIONS) == 50, f"PREDEFINED_REGIONS should have 50 items, got {len(PREDEFINED_REGIONS)}"
    assert len(TRAVEL_TIME_SLOTS) == 6, f"TRAVEL_TIME_SLOTS should have 6 items, got {len(TRAVEL_TIME_SLOTS)}"
    print(f"  ✅ List lengths validated")

    # Validate dict structures
    expected_scenario_categories = {'general', 'restaurant', 'hotel', 'train', 'attraction', 'taxi', 'hospital', 'bus', 'flight'}
    assert set(SCENARIO_CATEGORIES.keys()) == expected_scenario_categories, f"SCENARIO_CATEGORIES keys mismatch"
    print(f"  ✅ Scenario categories structure validated")

    # Validate resolution status probabilities sum to 1.0
    resolution_sum = sum(RESOLUTION_STATUSES.values())
    assert abs(resolution_sum - 1.0) < 0.001, f"Resolution statuses should sum to 1.0, got {resolution_sum}"
    print(f"  ✅ Resolution status probabilities validated")

    # Validate service weights sum to 1.0
    service_weights_sum = sum(SERVICE_WEIGHTS.values())
    assert abs(service_weights_sum - 1.0) < 0.001, f"Service weights should sum to 1.0, got {service_weights_sum}"
    print(f"  ✅ Service weights validated")

    # Validate logical combinations structure
    assert 'double' in LOGICAL_COMBINATIONS, "Missing 'double' in LOGICAL_COMBINATIONS"
    assert 'triple' in LOGICAL_COMBINATIONS, "Missing 'triple' in LOGICAL_COMBINATIONS"
    assert 'quadruple' in LOGICAL_COMBINATIONS, "Missing 'quadruple' in LOGICAL_COMBINATIONS"
    print(f"  ✅ Logical combinations structure validated")

    # Validate all services in combinations are from CORE_SERVICES
    all_services_in_combinations = set()
    for combination_type, combinations in LOGICAL_COMBINATIONS.items():
        for combination in combinations:
            all_services_in_combinations.update(combination)

    invalid_services = all_services_in_combinations - set(CORE_SERVICES)
    assert len(invalid_services) == 0, f"Invalid services in combinations: {invalid_services}"
    print(f"  ✅ All services in combinations are valid")

    if all_passed:
        print("\n🎉 ALL CONSTANTS VALIDATION PASSED!")
        print("✅ Constants extracted successfully with 100% accuracy")
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
    print(f"Assistant emotions: {len(ASSISTANT_EMOTION_LIST)} items")
    print(f"Scenario categories: {len(SCENARIO_CATEGORIES)} service types")
    print(f"Predefined regions: {len(PREDEFINED_REGIONS)} cities")
    print(f"Travel time slots: {len(TRAVEL_TIME_SLOTS)} periods")
    print(f"Resolution statuses: {len(RESOLUTION_STATUSES)} types")
    print(f"Core services: {len(CORE_SERVICES)} services")
    print(f"Service combinations:")
    for combo_type, combos in LOGICAL_COMBINATIONS.items():
        print(f"  - {combo_type}: {len(combos)} combinations")
    print("=" * 50)

if __name__ == "__main__":
    try:
        success = test_constants_extraction()
        if success:
            show_constants_summary()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)