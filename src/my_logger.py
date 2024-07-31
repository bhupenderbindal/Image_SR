import logging

class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._is_initialized = False
        return cls._instance

    def initialize(self, name, level=logging.INFO, log_file="app.log"):
        if not self._is_initialized:
            self.logger = logging.getLogger(name)
            self.logger.setLevel(level)
            
            # Create file handler
            file_handler = logging.FileHandler(log_file, mode='w')
            file_handler.setLevel(level)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            # Define logger format
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            # Add handlers to the logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

            self._is_initialized = True

    def get_logger(self):
        if not self._is_initialized:
            raise RuntimeError("Logger is not initialized. Call initialize() before using get_logger().")
        return self.logger

# Initialize logger in main file like this:
# logger_instance = Logger()
# logger_instance.initialize('my_app')
# logger = logger_instance.get_logger()
