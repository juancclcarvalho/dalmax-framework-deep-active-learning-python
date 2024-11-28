class Foo:
    def __init__(self):
        self.bar = 42

    def get_bar(self):
        return self.bar

    def set_bar(self, value):
        self.bar = value

foo = Foo()
print(foo.get_bar())

foo2 = Foo
print(foo2.get_bar(foo2))