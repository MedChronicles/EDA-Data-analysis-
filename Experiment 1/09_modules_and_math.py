import math
import random
import datetime

print(math.sqrt(64))
print(math.pi)
print(math.floor(4.7))
print(math.ceil(4.2))

print(random.randint(1, 10))
numbers = [1, 2, 3, 4, 5]
print(random.choice(numbers))
random.shuffle(numbers)
print(numbers)

today = datetime.date.today()
print(today)
print(today.year, today.month, today.day)
