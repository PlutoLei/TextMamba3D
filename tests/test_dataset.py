# tests/test_dataset.py
import torch
import pytest


class TestTextGenerator:
    def test_generate_diagnosis_text(self):
        """Test diagnosis text generation from mask."""
        from data.text_generator import DiagnosisTextGenerator

        generator = DiagnosisTextGenerator()

        # Create fake mask
        mask = torch.zeros(96, 96, 96, dtype=torch.long)
        mask[40:60, 40:60, 40:60] = 1  # Necrotic
        mask[35:65, 35:65, 35:65] = 2  # Edema (surrounding)
        mask[45:55, 45:55, 45:55] = 4  # Enhancing (core)

        text = generator.generate(mask)

        assert isinstance(text, str)
        assert len(text) > 10
