import subprocess
import sys


def test_viz_does_not_import_napari_at_import_time():
    # napari is an optional extra, so importing the module must not need it
    code = "import sys, spleleton.viz; assert 'napari' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
