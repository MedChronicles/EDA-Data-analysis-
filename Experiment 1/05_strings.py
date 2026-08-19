text = "  Python is Powerful, Python is Elegant!  "

print(text.strip())
print(text.lower().strip())
print(text.upper().strip())
print(text.replace("Python", "Java").strip())
print(text.split())
print(text.count("Python"))

words = ["Data", "Science", "with", "Python"]
print(" ".join(words))
print("-".join(words))

sample = "Hello, World!"
print(sample[:5])
print(sample[-6:])
print(sample[::-1])

name = "Sara"
score = 92.5
print(f"{name} scored {score:.1f}")
print("{} scored {}".format(name, score))
