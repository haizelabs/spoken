#!/usr/bin/env python3
"""
Test script to verify temperature inference works correctly.
"""

import inspect
from speak.models.openai import OpenAISpeechToSpeechHarness
from speak.models.nova import NovaSpeechToSpeechHarness
from speak.models.gemini import GeminiSpeechToSpeechHarness

def get_default_temperature(cls):
    """Get the default temperature from a class's __init__ method."""
    init_sig = inspect.signature(cls.__init__)
    temperature_param = init_sig.parameters.get('temperature')
    if temperature_param and temperature_param.default != inspect.Parameter.empty:
        return temperature_param.default
    return None

def test_temperature_inference():
    """Test that each model has the expected default temperature."""
    
    # Test OpenAI
    openai_temp = get_default_temperature(OpenAISpeechToSpeechHarness)
    print(f"OpenAI default temperature: {openai_temp}")
    assert openai_temp == 0.8, f"Expected 0.8, got {openai_temp}"
    
    # Test Nova
    nova_temp = get_default_temperature(NovaSpeechToSpeechHarness)
    print(f"Nova default temperature: {nova_temp}")
    assert nova_temp == 0.7, f"Expected 0.7, got {nova_temp}"
    
    # Test Gemini
    gemini_temp = get_default_temperature(GeminiSpeechToSpeechHarness)
    print(f"Gemini default temperature: {gemini_temp}")
    assert gemini_temp == 1.0, f"Expected 1.0, got {gemini_temp}"
    
    print("✅ All temperature defaults are correct!")

if __name__ == "__main__":
    test_temperature_inference() 