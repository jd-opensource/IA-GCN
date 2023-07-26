import os
import sys

python_path=os.popen("which python").read().strip()
if not "anaconda" in python_path:
    if os.path.exists("/usr/local/anaconda2"):
        python_path = "/usr/local/anaconda2/bin/python"
        os.system("%s Light_GCN_ops.py %s"%(python_path, " ".join(sys.argv[1:])))
    else:
        python_path = "/usr/local/anaconda3/bin/python"
        os.system("%s Light_GCN_ops.py %s"%(python_path, " ".join(sys.argv[1:])))
