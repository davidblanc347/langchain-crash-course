
import os

# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "moby_dick.txt")
# Test reading the file directly
try:
    with open(file_path, 'r',encoding='utf-8') as test_file:
        print("First 100 characters of the file:")
        print(test_file.read(100))
except Exception as e:
    raise RuntimeError(f"Failed to read the file due to: {e}")