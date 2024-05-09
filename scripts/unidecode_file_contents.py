"""A script to convert file contents to ASCII. Necessary to use old splits after using
`unidecode` to save file names in chord tones data.
"""

import os
import argparse
from unidecode import unidecode


def process_files(root_dir, dry_run=False, encoding="utf-8"):
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding=encoding) as file:
                    original_content = file.read()
                new_content = unidecode(original_content)

                if original_content != new_content:
                    if dry_run:
                        print(f"Would convert '{file_path}'")
                    else:
                        with open(file_path, "w", encoding=encoding) as file:
                            file.write(new_content)
                        print(f"Converted '{file_path}'")
            except UnicodeDecodeError:
                print(f"Skipping file due to encoding issues: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert file contents to ASCII.")
    parser.add_argument("directory", type=str, help="Directory to process files in")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without modifying any files",
    )

    args = parser.parse_args()

    process_files(args.directory, dry_run=args.dry_run)
