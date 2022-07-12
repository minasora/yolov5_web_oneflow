pip config set global.index -url https://pypi.tuna.tsing.edu.cn/simple
pip install -r requirements.txt
python -m pip uninstall -y oneflow
python -m pip install -f https://staging.oneflow.info/branch/master/cu112 --pre oneflow
