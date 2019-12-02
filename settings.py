import logging
import os

log_level = logging.DEBUG
log_file_name = 'test_log_file.log'
log_file_path = os.path.dirname(__file__) + "\\" + log_file_name
logging.basicConfig(filename=log_file_path, level= log_level)
logger = logging.getLogger()

logging.getLogger("imutils").setLevel(logging.INFO)
logging.getLogger("cv2").setLevel(logging.INFO)
logging.getLogger("flask").setLevel(logging.INFO)
logging.getLogger("keras").setLevel(logging.INFO)
logging.getLogger("tkinter").setLevel(logging.INFO)
logging.getLogger("webbrowser").setLevel(logging.INFO)
logging.getLogger("matplotib").setLevel(logging.WARNING)