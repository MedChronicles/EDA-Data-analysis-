class Animal:
    def __init__(self, name, sound):
        self.name = name
        self.sound = sound

    def make_sound(self):
        return f"{self.name} says {self.sound}"

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name, "Woof")
        self.breed = breed

    def fetch(self):
        return f"{self.name} the {self.breed} fetches the ball"

class Cat(Animal):
    def __init__(self, name):
        super().__init__(name, "Meow")

dog = Dog("Rex", "Labrador")
cat = Cat("Whiskers")

print(dog.make_sound())
print(dog.fetch())
print(cat.make_sound())

print(isinstance(dog, Animal))
print(issubclass(Dog, Animal))

class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        self.balance += amount

    def withdraw(self, amount):
        if amount > self.balance:
            print("Insufficient funds")
        else:
            self.balance -= amount

account = BankAccount("Dana", 100)
account.deposit(50)
account.withdraw(30)
print(account.balance)
