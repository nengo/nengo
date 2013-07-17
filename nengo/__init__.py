import logging

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger('nengo')
logger.setLevel(logging.DEBUG)

from .model import Model
