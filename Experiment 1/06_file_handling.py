import os

os.makedirs("demo_files", exist_ok=True)
text_path = "demo_files/notes.txt"

with open(text_path, "w") as f:
    f.write("Line 1\n")
    f.write("Line 2\n")
    f.write("Line 3\n")

with open(text_path, "r") as f:
    for line in f:
        print(line.strip())

with open(text_path, "a") as f:
    f.write("Line 4\n")

with open(text_path, "r") as f:
    print(f.read())
