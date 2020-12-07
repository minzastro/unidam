
class DummyLog():
    def info(self, value):
        print(value)

    def warn(self, value):
        print(value)

    def debug(self, value):
        print(value)

def get_logger(name, screen_output=True, log_filename=''):
    return DummyLog()