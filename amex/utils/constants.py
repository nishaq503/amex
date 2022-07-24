import logging
import os

KAGGLE_LOG = getattr(logging, os.environ.get('KAGGLE_LOG', 'INFO'))

MAX_WORKERS = max(1, int(os.cpu_count() / 2))
EPSILON = 1e-8  # To avoid divide-by-zero errors


class Unset:
    """ This is a hack around type-hinting when a value cannot be set in the
    __init__ method for a class.
    https://peps.python.org/pep-0661/
    Usage:
    ```python
    class MyClass:
        def __init__(self, *args, **kwargs):
            ...
            self.__value: typing.Union[ValueType, Unset] = UNSET
        def value_setter(self, *args, **kwargs):
            ...
            self.__value = something
            return
        @property
        def value(self) -> ValueType:
            if self.__value is UNSET:
                raise ValueError(f'Please call `value_setter` on the object before using this property.')
            return self.__value
    ```
    """
    __unset = None

    def __new__(cls):
        if cls.__unset is None:
            cls.__unset = super().__new__(cls)
        return cls.__unset


UNSET = Unset()
