import spleleton


def test_imports_with_version():
    assert isinstance(spleleton.__version__, str)
