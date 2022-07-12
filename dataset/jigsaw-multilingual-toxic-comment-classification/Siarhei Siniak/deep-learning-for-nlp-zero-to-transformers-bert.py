import os
assert os.system(r'''
    rm -fr repo
    git clone https://github.com/nartes/freelance-project-34-marketing-blog repo
    pip install -r repo/requirements.txt
''') == 0
import sys
sys.path.append('repo')
import python.tasks.jigsaw_toxic
import importlib
importlib.reload(python.tasks.jigsaw_toxic)
o_5 = python.tasks.jigsaw_toxic.kernel_5()