def describe_pet(name, species="dog"):
    print(name, "is a", species)

describe_pet("Rex")
describe_pet("Whiskers", "cat")

def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3, 4))

def show_info(**kwargs):
    for key, value in kwargs.items():
        print(key, value)

show_info(age=3, color="black")

square = lambda x: x ** 2
print(square(9))

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(6))
