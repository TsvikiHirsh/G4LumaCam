#!/usr/bin/env python3
"""
Basic validation test for detector models.
This script checks that:
1. DetectorModel enum is accessible
2. All detector models can be instantiated
3. Helper methods have correct signatures
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from lumacam.optics import Lens, DetectorModel, VerbosityLevel

def test_detector_model_enum():
    """Test that DetectorModel enum has all expected values."""
    print("Testing DetectorModel enum...")
    expected_models = [
        'IMAGE_INTENSIFIER',
        'GAUSSIAN_DIFFUSION',
        'DIRECT_DETECTION',
        'WAVELENGTH_DEPENDENT',
        'AVALANCHE_GAIN'
    ]

    for model_name in expected_models:
        assert hasattr(DetectorModel, model_name), f"Missing model: {model_name}"
        model = getattr(DetectorModel, model_name)
        print(f"  ✓ {model_name} = {model}")

    print("✓ All detector models present\n")

def test_helper_methods():
    """Test that all helper methods exist."""
    print("Testing helper methods...")
    lens = Lens(archive="/tmp/test_detector_models")

    methods = [
        '_apply_image_intensifier_model',
        '_apply_gaussian_diffusion_model',
        '_apply_direct_detection_model',
        '_apply_wavelength_dependent_model',
        '_apply_avalanche_gain_model'
    ]

    for method_name in methods:
        assert hasattr(lens, method_name), f"Missing method: {method_name}"
        print(f"  ✓ {method_name}")

    print("✓ All helper methods present\n")

def test_basic_model_invocation():
    """Test that models can be called without errors."""
    print("Testing basic model invocation...")
    lens = Lens(archive="/tmp/test_detector_models")

    # Test each model's helper method
    cx, cy, toa = 100.5, 100.5, 1000.0

    # IMAGE_INTENSIFIER
    result = lens._apply_image_intensifier_model(cx, cy, toa, blob=2.0, blob_variance=0.5, decay_time=100.0)
    assert len(result) == 4, "IMAGE_INTENSIFIER should return 4 values"
    print("  ✓ IMAGE_INTENSIFIER")

    # GAUSSIAN_DIFFUSION
    result = lens._apply_gaussian_diffusion_model(cx, cy, toa, sigma=1.5, model_params={})
    assert len(result) == 4, "GAUSSIAN_DIFFUSION should return 4 values"
    print("  ✓ GAUSSIAN_DIFFUSION")

    # DIRECT_DETECTION
    result = lens._apply_direct_detection_model(cx, cy, toa)
    assert len(result) == 4, "DIRECT_DETECTION should return 4 values"
    print("  ✓ DIRECT_DETECTION")

    # WAVELENGTH_DEPENDENT
    wavelength = 500.0
    result = lens._apply_wavelength_dependent_model(cx, cy, toa, wavelength, blob=2.0, decay_time=100.0, model_params={})
    assert result is None or len(result) == 4, "WAVELENGTH_DEPENDENT should return None or 4 values"
    print("  ✓ WAVELENGTH_DEPENDENT")

    # AVALANCHE_GAIN
    result = lens._apply_avalanche_gain_model(cx, cy, toa, blob=0.5, model_params={})
    assert len(result) == 5, "AVALANCHE_GAIN should return 5 values"
    print("  ✓ AVALANCHE_GAIN")

    print("✓ All models can be invoked\n")

def test_saturate_photons_signature():
    """Test that saturate_photons has the correct signature."""
    print("Testing saturate_photons signature...")
    lens = Lens(archive="/tmp/test_detector_models")

    # Check method exists
    assert hasattr(lens, 'saturate_photons')

    # Create minimal test data
    test_data = pd.DataFrame({
        'pixel_x': [100.0, 101.0, 102.0],
        'pixel_y': [100.0, 101.0, 102.0],
        'toa2': [100.0, 200.0, 300.0],
        'id': [1, 2, 3],
        'neutron_id': [1, 1, 1],
        'pulse_id': [1, 1, 1],
        'pulse_time_ns': [0, 0, 0],
        'wavelength': [500.0, 500.0, 500.0]
    })

    # Test with different models
    for model in DetectorModel:
        print(f"  Testing {model.name}...", end=" ")
        try:
            result = lens.saturate_photons(
                data=test_data,
                detector_model=model,
                blob=1.0,
                deadtime=600.0,
                seed=42,
                verbosity=VerbosityLevel.QUIET
            )
            print("✓")
        except Exception as e:
            print(f"✗ Error: {e}")
            raise

    print("✓ saturate_photons works with all models\n")

def main():
    """Run all tests."""
    print("=" * 60)
    print("Detector Models Validation Tests")
    print("=" * 60 + "\n")

    try:
        test_detector_model_enum()
        test_helper_methods()
        test_basic_model_invocation()
        test_saturate_photons_signature()

        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print("TEST FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
