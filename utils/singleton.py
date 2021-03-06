# encoding: utf-8


import threading


class MetaSingleton(type):
    """Singleton metaclass"""
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with MetaSingleton._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instance
