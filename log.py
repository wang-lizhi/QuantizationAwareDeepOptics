from datetime import datetime


class Logger:
    @staticmethod
    def i(content, *args):
        print("{} [INFO] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), content), *args)

    @staticmethod
    def e(content, *args):
        print("{} [ERROR] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), content), *args)

    @staticmethod
    def w(content, *args):
        print("{} [WARNING] [!] {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), content), *args)
