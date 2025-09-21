#!/usr/bin/env python3
"""
Validation test for time generator extraction
Ensures time generation produces correct format and handles edge cases
"""

import sys
import os
import random

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_time_generator():
    """Test extracted TimeGenerator produces valid results"""

    print("🔍 Validating time generator extraction...")

    from dialogue_generation.utils.time_generator import TimeGenerator, SecureTimeGenerator
    from dialogue_generation.config.constants import TRAVEL_TIME_SLOTS

    # Test both regular and secure versions
    generators = [
        ("Regular", TimeGenerator()),
        ("Secure", SecureTimeGenerator())
    ]

    all_passed = True

    for gen_name, time_gen in generators:
        print(f"  🧪 Testing {gen_name} TimeGenerator...")

        # Test with all time slots
        for slot in TRAVEL_TIME_SLOTS:
            print(f"    🕐 Testing slot: {slot[2]} ({slot[0]:02d}:00 - {slot[1]:02d}:00)")

            # Test multiple generations to ensure consistency
            times_generated = []
            for i in range(10):
                # Set seed for reproducible test (where applicable)
                if gen_name == "Regular":
                    random.seed(12345 + i)

                generated_time = time_gen.generate_random_time(slot)
                times_generated.append(generated_time)

                # Validate time format
                if not validate_time_format(generated_time):
                    print(f"      ❌ Invalid time format: {generated_time}")
                    all_passed = False
                    continue

                # Validate time is within slot (for normal slots)
                if not validate_time_in_slot(generated_time, slot):
                    print(f"      ❌ Time {generated_time} not in slot {slot}")
                    all_passed = False
                    continue

            print(f"      ✅ Generated valid times: {times_generated[:3]}... (sample)")

        # Test error handling with invalid slots
        invalid_slots = [
            (25, 30, "Invalid"),  # Invalid hours
            (-1, 5, "Negative"),  # Negative hour
            (12, 25, "Invalid End"),  # Invalid end hour
        ]

        print(f"    ⚠️ Testing error handling for {gen_name}...")
        for invalid_slot in invalid_slots:
            result = time_gen.generate_random_time(invalid_slot)
            if result != "00:00":
                print(f"      ❌ Should return '00:00' for invalid slot {invalid_slot}, got {result}")
                all_passed = False
            else:
                print(f"      ✅ Correctly handled invalid slot: {invalid_slot}")

    # Test deterministic behavior for regular generator
    print("  🔁 Testing deterministic behavior...")
    regular_gen = TimeGenerator()
    test_slot = (9, 17, "Test")

    # Test that same seed produces same result
    random.seed(12345)
    result1 = regular_gen.generate_random_time(test_slot)
    random.seed(12345)
    result2 = regular_gen.generate_random_time(test_slot)

    if result1 == result2:
        print(f"    ✅ Deterministic behavior confirmed: {result1}")
    else:
        print(f"    ❌ Non-deterministic behavior: {result1} vs {result2}")
        all_passed = False

    # Test secure generator produces different results (non-deterministic)
    print("  🔒 Testing secure random behavior...")
    secure_gen = SecureTimeGenerator()
    secure_results = set()

    for i in range(20):
        result = secure_gen.generate_random_time(test_slot)
        secure_results.add(result)

    if len(secure_results) > 1:
        print(f"    ✅ Secure generator produces varied results: {len(secure_results)} unique times")
    else:
        print(f"    ⚠️ Secure generator produced only {len(secure_results)} unique times (may be normal)")

    if all_passed:
        print("\n🎉 TIME GENERATOR VALIDATION PASSED!")
        print("✅ Time generator extracted successfully - safe to proceed")
        return True
    else:
        print("\n❌ TIME GENERATOR VALIDATION FAILED!")
        print("🚨 DO NOT PROCEED - Fix time generator first")
        return False

def validate_time_format(time_str: str) -> bool:
    """Validate time string is in HH:MM format"""
    if len(time_str) != 5:
        return False
    if time_str[2] != ':':
        return False

    try:
        hour_str, minute_str = time_str.split(':')
        hour = int(hour_str)
        minute = int(minute_str)

        # Validate ranges
        if not (0 <= hour <= 23):
            return False
        if not (0 <= minute <= 59):
            return False

        # Validate 5-minute intervals
        if minute % 5 != 0:
            return False

        return True
    except ValueError:
        return False

def validate_time_in_slot(time_str: str, slot: tuple) -> bool:
    """Validate time is within the given slot (allowing for overnight slots)"""
    start_hour, end_hour, _ = slot

    try:
        hour, minute = map(int, time_str.split(':'))
        time_minutes = hour * 60 + minute

        # Handle overnight slots
        if end_hour < start_hour:
            # Overnight slot: valid if >= start_hour OR < end_hour
            start_minutes = start_hour * 60
            end_minutes = end_hour * 60

            return (time_minutes >= start_minutes) or (time_minutes < end_minutes)
        else:
            # Normal slot: valid if between start and end (end exclusive like randint)
            start_minutes = start_hour * 60
            end_minutes = end_hour * 60

            # Allow end time since randint is end_minutes - 1, so it can generate end_minutes - 1
            # which when rounded up can become end_minutes
            return start_minutes <= time_minutes <= end_minutes

    except ValueError:
        return False

if __name__ == "__main__":
    try:
        success = test_time_generator()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)