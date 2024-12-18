from pathlib import Path


def delete_dir(directory: Path):
    """
    deletes `directory` including all sub-directories and all files
    """
    assert directory.is_dir(), "input path is not a directory"
    for subdir in directory.iterdir():
        if subdir.is_file():
            subdir.unlink()
            continue
        delete_dir(subdir)
    directory.rmdir()
