fruits = ["apple", "banana", "cherry"]
fruits.append("date")
fruits.insert(1, "avocado")
print(fruits)
print(fruits[1:3])
print(fruits[::-1])

coordinates = (10.0, 20.0)
x, y = coordinates
print(x, y)

a_set = {1, 2, 3, 4}
b_set = {3, 4, 5, 6}
print(a_set | b_set)
print(a_set & b_set)
print(a_set - b_set)

student = {"name": "Bob", "age": 22, "major": "CS"}
student["gpa"] = 3.8
for key, value in student.items():
    print(key, value)

squares = [n ** 2 for n in range(1, 11)]
print(squares)

evens_only = [n for n in range(20) if n % 2 == 0]
print(evens_only)

word_lengths = {word: len(word) for word in fruits}
print(word_lengths)
