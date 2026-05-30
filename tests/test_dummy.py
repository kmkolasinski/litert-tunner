from litert_tunner import __version__


def test_dummy():
    assert True


def test_version_is_string():
    assert isinstance(__version__, str)
