import unicodedata

def find_non_ascii_chars(filepath):
    non_ascii_chars = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        for char in content:
            if ord(char) > 127:
                non_ascii_chars.add(char)

    if not non_ascii_chars:
        print("No non-ASCII characters found.")
        return

    print("Found the following unique non-ASCII characters:")
    for char in sorted(list(non_ascii_chars)):
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "No name found"
        print(f"Character: '{char}', Code Point: U+{ord(char):04X}, Name: {name}")

if __name__ == "__main__":
    find_non_ascii_chars("failing.txt")
