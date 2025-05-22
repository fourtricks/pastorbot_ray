def save_text_to_file(text, filename):
    """
    Saves the given text to a .txt file with the provided filename.

    :param text: The text content to write.
    :param filename: The desired name of the file (without .txt extension).
    """
    full_filename = f"{filename}.txt"
    try:
        with open(full_filename, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"File saved as '{full_filename}'")
    except Exception as e:
        print(f"An error occurred: {e}")


# 🔹 Paste your content and desired file name below
text_to_save = """
insert text here
"""

file_name = "insert title here"  # No .txt extension needed

save_text_to_file(text_to_save, file_name)
