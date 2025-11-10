from flask import Flask, request, jsonify, send_file
import torch
import yaml
import numpy as np
import io
import soundfile as sf
import logging
from pathlib import Path

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class UkrainianTTS:
    def __init__(self, model_path="models"):
        self.model_path = Path(model_path)
        self.model = None
        self.config = None
        self.load_model()
    
    def load_model(self):
        """Завантажуємо модель Oleksa"""
        try:
            # Завантажуємо конфігурацію
            with open(self.model_path / 'config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
            
            # Завантажуємо модель
            self.model = torch.load(
                self.model_path / 'model.pth', 
                map_location='cpu',
                weights_only=False
            )
            
            # Завантажуємо статистику
            self.feats_stats = np.load(self.model_path / 'feats_stats.npz')
            
            logging.info("✅ Ukrainian TTS модель завантажена")
            
        except Exception as e:
            logging.error(f"❌ Помилка завантаження моделі: {e}")
            raise
    
    def synthesize(self, text):
        """Синтезує мову з тексту"""
        try:
            # Тут буде ваша логіка синтезу з моделлю
            # Для прикладу - створюємо простий аудіо
            
            sample_rate = 22050
            duration = max(1.0, len(text) * 0.15)
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            base_freq = 180  # Чоловічий голос
            
            # Створюємо складніший звук
            audio = np.zeros_like(t)
            for harmonic in range(1, 6):
                freq = base_freq * harmonic
                amplitude = 0.5 / harmonic
                audio += amplitude * np.sin(2 * np.pi * freq * t)
            
            # Додаємо модуляцію для ефекту мови
            modulation = 0.3 * np.sin(2 * np.pi * 5 * t)
            audio *= (1 + modulation)
            
            # Нормалізуємо
            audio = audio / np.max(np.abs(audio)) * 0.8
            
            return audio, sample_rate
            
        except Exception as e:
            logging.error(f"❌ Помилка синтезу: {e}")
            return None, None

# Ініціалізація TTS
tts = UkrainianTTS()

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """API для синтезу мови"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'Текст відсутній'}), 400
        
        if len(text) > 1000:
            return jsonify({'error': 'Текст занадто довгий'}), 400
        
        logging.info(f"🎯 Синтез тексту: '{text}'")
        
        # Синтезуємо аудіо
        audio, sample_rate = tts.synthesize(text)
        
        if audio is not None:
            # Зберігаємо в буфер
            buffer = io.BytesIO()
            sf.write(buffer, audio, sample_rate, format='WAV')
            buffer.seek(0)
            
            return send_file(
                buffer,
                mimetype='audio/wav',
                as_attachment=True,
                download_name='ukrainian_speech.wav'
            )
        else:
            return jsonify({'error': 'Помилка синтезу'}), 500
            
    except Exception as e:
        logging.error(f"❌ Помилка API: {e}")
        return jsonify({'error': 'Внутрішня помилка сервера'}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ready', 
        'voice': 'oleksa',
        'language': 'ukrainian'
    })

if __name__ == '__main__':
    print("🚀 Ukrainian TTS Server запускається...")
    print("🔊 Голос: Oleksa")
    print("🌐 API: http://localhost:5000/synthesize")
    app.run(host='0.0.0.0', port=5000, debug=False)