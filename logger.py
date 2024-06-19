import logging

class Logger:
    def __init__(self, name, log_path=None, stream=True):
        self.logger = logging.Logger(name)
        self.file = log_path is not None
        self.stream = stream
        formatter = logging.Formatter('%(asctime)s: %(message)s')
        if self.file:
            handler = logging.FileHandler(log_path)
            handler.setLevel(logging.INFO)
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        if self.stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setLevel(logging.INFO)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def log(self, content):
        if self.file or self.stream:
            self.logger.info(content)