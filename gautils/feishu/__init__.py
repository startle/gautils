from .tools import send_fs_robot_msg
from .core import *
import os

if not os.path.exists(DEBUG_OUT_DIR):
    os.makedirs(DEBUG_OUT_DIR)
