from abc import ABC, abstractmethod

class MyAbstractClass(ABC):
    def __init__(self, value):
        self.value = value
        super().__init__()
    
    @abstractmethod
    def abstract_method(self):
        pass

    def concrete_method(self):
        print("这是一个具体方法，可以在子类中直接使用")

class MyConcreteClass(MyAbstractClass):
    def __init__(self, value, extra):
        super().__init__(value)
        self.extra = extra

    def some_abstract_method(self):
        print(f"Value: {self.value}, Extra: {self.extra}")
    
    def abstract_method(self):
        print("实现了抽象方法")

# 实例化
my_instance = MyConcreteClass(10, "extra_data")
my_instance.abstract_method()  # 必须实现的抽象方法
my_instance.concrete_method()  # 继承自抽象基类的具体方法
my_instance.some_abstract_method()  # 测试init方法
