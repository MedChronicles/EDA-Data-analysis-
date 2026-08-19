def classify_number(n):
    if n > 0:
        return "positive"
    elif n < 0:
        return "negative"
    else:
        return "zero"

for num in [10, -5, 0]:
    print(num, classify_number(num))

for i in range(1, 16):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)

count = 0
while count < 10:
    count += 1
    if count % 2 == 0:
        continue
    if count > 7:
        break
    print(count)
