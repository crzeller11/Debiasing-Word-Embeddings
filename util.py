import os

PROJECT_PATH = os.path.realpath(os.path.dirname(__file__))

CORPORA_PATH = os.path.join(PROJECT_PATH, 'corpora')
MODELS_PATH = os.path.join(PROJECT_PATH, 'models')
DIRECTIONS_PATH = os.path.join(PROJECT_PATH, 'directions')
WORDS_PATH = os.path.join(PROJECT_PATH, 'words')


def make_project_dirs():
    """Create project directories if they don't exist."""
    for path in [CORPORA_PATH, MODELS_PATH, DIRECTIONS_PATH, WORDS_PATH]:
        if not os.path.exists(path):
            os.mkdir(path)


def list_files(path):
    """List files in a project directory.

    Arguments:
        path (str): The directory/path to list files from.

    Returns:
        List[str]: List of files in the directory.
    """
    return [
        os.path.join(path, file)
        for file in os.listdir(path)
        if not file.startswith('.')
    ]

make_project_dirs()
