from pathlib import Path
relative_root = Path(__file__).parent.parent.parent.resolve()
import sys
sys.path.append(fr"{relative_root}")
from StyleGAN import *
from .gan_inversion_model import *
from .model import *
