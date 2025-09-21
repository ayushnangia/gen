#!/usr/bin/env python3
"""
Validation test for dataset loader extraction
Tests structure and methods without requiring actual datasets
"""

import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def test_dataset_loader_structure():
    """Test dataset loader classes and methods exist with correct signatures"""

    print("🔍 Validating dataset loader structure...")

    from dialogue_generation.data.dataset_loader import (
        DatasetLoader, PersonaManager, DialogueProcessor, DataRepository
    )

    all_passed = True

    # Test DatasetLoader
    print("  📋 Testing DatasetLoader...")
    try:
        loader = DatasetLoader()

        # Check required methods exist
        required_methods = ['load_dataset', 'load_persona_dataset']
        for method_name in required_methods:
            if not hasattr(loader, method_name):
                print(f"    ❌ Missing method: {method_name}")
                all_passed = False
            elif not callable(getattr(loader, method_name)):
                print(f"    ❌ Method not callable: {method_name}")
                all_passed = False
            else:
                print(f"    ✅ Has method: {method_name}")

    except Exception as e:
        print(f"    ❌ DatasetLoader initialization failed: {e}")
        all_passed = False

    # Test PersonaManager (with mock data)
    print("  👤 Testing PersonaManager...")
    try:
        # Create mock persona dataset
        mock_persona_data = {
            'cluster_label': [1, 2, 1, 3, 2],
            'summary_label': ['["label1"]', '["label2"]', '["label1", "label3"]', '["label4"]', '["label2"]'],
            'persona': ['persona1', 'persona2', 'persona3', 'persona4', 'persona5']
        }

        # Create mock dataset object
        class MockPersonaDataset:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, idx):
                return {key: values[idx] for key, values in self.data.items()}

            def __len__(self):
                return len(self.data['persona'])

        mock_dataset = MockPersonaDataset(mock_persona_data)
        persona_manager = PersonaManager(mock_dataset)

        # Check required methods exist
        required_methods = ['safe_eval', 'get_summary_labels', 'populate_persona_dicts', 'select_random_persona']
        for method_name in required_methods:
            if not hasattr(persona_manager, method_name):
                print(f"    ❌ Missing method: {method_name}")
                all_passed = False
            elif not callable(getattr(persona_manager, method_name)):
                print(f"    ❌ Method not callable: {method_name}")
                all_passed = False
            else:
                print(f"    ✅ Has method: {method_name}")

        # Test persona selection
        persona = persona_manager.select_random_persona()
        if persona in mock_persona_data['persona']:
            print(f"    ✅ Persona selection works: {persona}")
        else:
            print(f"    ❌ Invalid persona selected: {persona}")
            all_passed = False

        # Test cluster and summary dicts populated
        if len(persona_manager.cluster_dict) > 0:
            print(f"    ✅ Cluster dict populated: {len(persona_manager.cluster_dict)} clusters")
        else:
            print(f"    ❌ Cluster dict not populated")
            all_passed = False

        if len(persona_manager.summary_dict) > 0:
            print(f"    ✅ Summary dict populated: {len(persona_manager.summary_dict)} summaries")
        else:
            print(f"    ❌ Summary dict not populated")
            all_passed = False

    except Exception as e:
        print(f"    ❌ PersonaManager testing failed: {e}")
        all_passed = False

    # Test DialogueProcessor
    print("  💬 Testing DialogueProcessor...")
    try:
        processor = DialogueProcessor()

        # Check required methods exist
        required_methods = ['extract_dialogue', 'generate_base_conversation']
        for method_name in required_methods:
            if not hasattr(processor, method_name):
                print(f"    ❌ Missing method: {method_name}")
                all_passed = False
            elif not callable(getattr(processor, method_name)):
                print(f"    ❌ Method not callable: {method_name}")
                all_passed = False
            else:
                print(f"    ✅ Has method: {method_name}")

        # Test with mock dialogue data
        mock_dialogue = {
            "dialogue_id": "test_001",
            "turns": {
                "speaker": [0, 1, 0, 1],
                "utterance": ["Hello", "Hi there!", "How are you?", "I'm doing well, thanks!"],
                "turn_id": [1, 2, 3, 4]
            }
        }

        # Test dialogue extraction
        turns = processor.extract_dialogue(mock_dialogue)
        if len(turns) == 4:
            print(f"    ✅ Dialogue extraction works: {len(turns)} turns")
        else:
            print(f"    ❌ Wrong number of turns extracted: {len(turns)}")
            all_passed = False

        # Check turn structure
        if turns and all(key in turns[0] for key in ['turn_id', 'speaker', 'utterance']):
            print(f"    ✅ Turn structure correct")
        else:
            print(f"    ❌ Turn structure incorrect: {turns[0] if turns else 'No turns'}")
            all_passed = False

        # Test base conversation generation
        base_conv = processor.generate_base_conversation(turns)
        if "USER:" in base_conv and "ASSISTANT:" in base_conv:
            print(f"    ✅ Base conversation generation works")
        else:
            print(f"    ❌ Base conversation format incorrect: {base_conv[:100]}...")
            all_passed = False

    except Exception as e:
        print(f"    ❌ DialogueProcessor testing failed: {e}")
        all_passed = False

    # Test DataRepository
    print("  🗃️ Testing DataRepository...")
    try:
        # We can't test actual loading without datasets, but we can test structure
        repo = DataRepository()

        # Check required methods exist
        required_methods = ['get_dataset_info', 'select_random_persona', 'extract_and_process_dialogue']
        for method_name in required_methods:
            if not hasattr(repo, method_name):
                print(f"    ❌ Missing method: {method_name}")
                all_passed = False
            elif not callable(getattr(repo, method_name)):
                print(f"    ❌ Method not callable: {method_name}")
                all_passed = False
            else:
                print(f"    ✅ Has method: {method_name}")

        # Check properties exist
        required_properties = ['dataset_loader', 'dialogue_processor']
        for prop_name in required_properties:
            if not hasattr(repo, prop_name):
                print(f"    ❌ Missing property: {prop_name}")
                all_passed = False
            else:
                print(f"    ✅ Has property: {prop_name}")

    except Exception as e:
        print(f"    ❌ DataRepository testing failed: {e}")
        all_passed = False

    # Test utility functions
    print("  🔧 Testing utility functions...")
    try:
        # Test safe_eval function
        manager = PersonaManager({})  # Empty for testing utilities

        # Test valid list string
        result = manager.safe_eval('["item1", "item2"]')
        if result == ["item1", "item2"]:
            print(f"    ✅ safe_eval works for valid list")
        else:
            print(f"    ❌ safe_eval failed for valid list: {result}")
            all_passed = False

        # Test invalid string
        result = manager.safe_eval('invalid syntax')
        if result == []:
            print(f"    ✅ safe_eval handles invalid syntax")
        else:
            print(f"    ❌ safe_eval didn't handle invalid syntax: {result}")
            all_passed = False

        # Test get_summary_labels
        result = manager.get_summary_labels('["label1", "label2"]')
        if result == ('label1', 'label2'):
            print(f"    ✅ get_summary_labels works")
        else:
            print(f"    ❌ get_summary_labels failed: {result}")
            all_passed = False

    except Exception as e:
        print(f"    ❌ Utility function testing failed: {e}")
        all_passed = False

    if all_passed:
        print("\n🎉 DATASET LOADER STRUCTURE VALIDATION PASSED!")
        print("✅ All classes and methods properly extracted - safe to proceed")
        return True
    else:
        print("\n❌ DATASET LOADER STRUCTURE VALIDATION FAILED!")
        print("🚨 DO NOT PROCEED - Fix dataset loader structure first")
        return False

def show_dataset_loader_summary():
    """Show summary of dataset loader structure"""
    from dialogue_generation.data.dataset_loader import (
        DatasetLoader, PersonaManager, DialogueProcessor, DataRepository
    )

    print("\n📊 DATASET LOADER SUMMARY:")
    print("=" * 50)
    print("Classes extracted:")
    print("  - DatasetLoader: Handles Multi-WOZ and persona dataset loading")
    print("  - PersonaManager: Manages persona selection and clustering")
    print("  - DialogueProcessor: Processes dialogue extraction and formatting")
    print("  - DataRepository: Central data management hub")
    print("\nKey methods available:")
    print("  - load_dataset(): Load Multi-WOZ dataset")
    print("  - load_persona_dataset(): Load persona dataset")
    print("  - extract_dialogue(): Extract turns from dialogue JSON")
    print("  - generate_base_conversation(): Format turns into conversation")
    print("  - select_random_persona(): Select random persona")
    print("=" * 50)

if __name__ == "__main__":
    try:
        success = test_dataset_loader_structure()
        if success:
            show_dataset_loader_summary()
        exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)