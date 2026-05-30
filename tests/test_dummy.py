from litert_tunner import __version__


def test_version_is_string():
    assert isinstance(__version__, str)


def test_version_is_not_empty():
    assert __version__
