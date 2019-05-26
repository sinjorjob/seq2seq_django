import os
from django.conf import settings

NUM_SAMPLES = 10000  # サンプル数
S2S_MODEL = os.path.join(settings.BASE_DIR, 'models\s2s.h5')
ENCODER_MODEL = os.path.join(settings.BASE_DIR, 'models\encoder_model.h5')
DECODER_MODEL = os.path.join(settings.BASE_DIR, 'models\decoder_model.h5')
DATA_PATH = os.path.join(settings.BASE_DIR, 'models\jpn.txt')
