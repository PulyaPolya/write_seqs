import os
import tempfile
import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--keep-files",
        action="store_true",
        help="keep temporary files for inspection",
    )
    parser.addoption(
        "--slow", action="store_true", help="run 'slow', comprehensive tests"
    )
    # parser.addoption(
    #     "--provided-paths",
    #     nargs="+",
    #     type=str,
    #     # action="store_true",
    #     help="specific paths on which to run tests (if applicable)",
    # )


@pytest.fixture(scope="session")
def slow(request):
    return request.config.option.slow


@pytest.fixture(scope="session")
def folders(keep_files):
    if not keep_files:
        directories = {
            "DST_DATA_DIR": tempfile.mkdtemp(),
            "DATASET_CACHE_DIR": tempfile.mkdtemp(),
        }
        yield directories
        for directory in directories.values():
            shutil.rmtree(directory)

    else:
        directories = {
            "DST_DATA_DIR": os.path.join(
                os.path.dirname((os.path.realpath(__file__))), "temp_out"
            ),
            "DATASET_CACHE_DIR": os.path.join(
                os.path.dirname((os.path.realpath(__file__))), "cache_out"
            ),
        }
        yield directories


@pytest.fixture(scope="session")
def keep_files(request):
    return request.config.option.keep_files


# @pytest.fixture(scope="session")
# def provided_paths(request):
#     return request.config.option.provided_paths


@pytest.fixture(autouse=True)
def mock_env_user(monkeypatch, folders):
    monkeypatch.setenv("DST_DATA_DIR", folders["DST_DATA_DIR"])
    monkeypatch.setenv("DATASET_CACHE_DIR", folders["DATASET_CACHE_DIR"])
    # monkeypatch.setenv("N_MIDI_FILES", os.environ.get("N_MIDI_FILES", "10"))
