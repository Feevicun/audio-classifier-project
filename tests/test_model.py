import pytest
import torch
import json
import os
from train import AudioClassifier, mel_spectrogram

class TestTrainedModel:
    @pytest.fixture
    def model(self):
        """Завантажуємо навчену модель"""
        if not os.path.exists('model.pth'):
            pytest.skip("Model not found - training might have failed")
        
        with open('class_info.json', 'r') as f:
            class_info = json.load(f)
        
        model = AudioClassifier(num_classes=len(class_info['target_classes']))
        model.load_state_dict(torch.load('model.pth', map_location='cpu'))
        model.eval()
        return model
    
    @pytest.fixture
    def class_info(self):
        with open('class_info.json', 'r') as f:
            return json.load(f)
    
    def test_model_loading(self, model):
        """Тест завантаження моделі"""
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_model_output_shape(self, model, class_info):
        """Тест форми виходу моделі"""
        num_classes = len(class_info['target_classes'])
        batch_size = 2
        channels = 1
        height = 64
        width = 32
        
        # Створюємо fake input
        dummy_input = torch.randn(batch_size, channels, height, width)
        
        with torch.no_grad():
            output = model(dummy_input)
        
        assert output.shape == (batch_size, num_classes)
    
    def test_class_info_structure(self, class_info):
        """Тест структури class_info"""
        assert 'target_classes' in class_info
        assert isinstance(class_info['target_classes'], list)
        assert len(class_info['target_classes']) > 0
    
    def test_training_log_exists(self):
        """Перевіряємо наявність логів тренування"""
        assert os.path.exists('training.log')
        
        with open('training.log', 'r') as f:
            log_content = f.read()
            assert len(log_content) > 0