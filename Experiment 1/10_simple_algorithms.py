def linear_search(items, target):
    for index, value in enumerate(items):
        if value == target:
            return index
    return -1

def bubble_sort(items):
    items = items.copy()
    n = len(items)
    for i in range(n):
        for j in range(n - i - 1):
            if items[j] > items[j + 1]:
                items[j], items[j + 1] = items[j + 1], items[j]
    return items

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

numbers = [5, 2, 9, 1, 5, 6]
print(linear_search(numbers, 9))
print(bubble_sort(numbers))
print([n for n in range(2, 30) if is_prime(n)])
