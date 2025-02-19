class Frame:
    # This class variable will be shared by all subclasses.
    shared_value = None

    @classmethod
    def set_shared_value(cls, value):
        cls.shared_value = value