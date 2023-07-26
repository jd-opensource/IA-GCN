#export PINO_USER=liuhu1
#pinoctl resources --cluster=langfang
#pinoctl resources --cluster=jdcloud-dev-zyx
#sleep 5
#pinoctl start --group=ads-search-ctr --timeout=59600 --storage=cfs .
set -x
export PINO_USER=liuhu1
#sh build.sh
#  --runtime=jointcnn-runtime2
#9nctl resources --cluster=kechuang
9Nctl start --email=liuhu1@jd.com --group=ea-ads-search-ctr --idle_timeout 24 .
