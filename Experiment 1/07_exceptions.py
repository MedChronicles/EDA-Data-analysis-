def divide(a, b):
    try:
        result = a / b
    except ZeroDivisionError:
        print("Cannot divide by zero")
        return None
    else:
        print("Division successful")
        return result
    finally:
        print("Done trying division")

print(divide(10, 2))
print(divide(10, 0))

class InvalidAgeError(Exception):
    pass

def check_age(age):
    if age < 0:
        raise InvalidAgeError("Age cannot be negative")
    return age

try:
    check_age(-5)
except InvalidAgeError as e:
    print(e)
