import logging

logging.basicConfig(format='[%(levelname)s] %(message)s')
logger = logging.getLogger('nengo')
logger.setLevel(logging.DEBUG)

__all__ = ['model', 'nonlinear', 'sim']

