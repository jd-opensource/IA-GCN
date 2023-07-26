# /bin/python
# encoding=utf-8

# python2.7
#framework="tensorflow-1.12.0-nm-cuda9-py2"   # 必选，指定训练环境
framework="9n-train:tf_1.12.0_gpu_ads_9ncloud_cuda9_cudnn7_py2_nm_bdpclient"
#framework="9n-train:tf_1.12.0_gpu_ads_9ncloud_cuda9.0_cudnn7_nm_bdpclient"
#cloud_tags={'cluster':'dev-cluster', 'node':'gpu9,gpu10,search8'}
#cloud_tags={'cluster':'dev-cluster', 'node':'gpu-online,search-online'}
cloud_tags={'cluster':'dev-cluster', 'node':'gpu1,gpu5,gpu6,train,search8'}
#cloud_tags={'node':'search1,search4,train','cluster':'jdcloud-dev-zyx'}

#如果需要使用大数据集市数据:
# 1. 需要确认镜像是否支持bdpclient，通过判断镜像名中是否存在`bdpclient`
# 2. 如果支持bdpclient需要添加以下配置，分别是hadoop集群、大数据集市名、用户账号、队列
# 3. 集市信息可以在堡垒机或者ea notebook终端环境变量中获取，环境变量对应关系如下：
#                   JDHXXXXX_USER -> hadoop_market
#                   JDHXXXXX_QUEUE -> hadoop_queue
#                   JDHXXXXX_CLUSTER_NAME -> hadoop_cluster
#                   TEAM_USER -> hadoop_user
hadoop_cluster="hope"
hadoop_market="mart_sch"
hadoop_user="ads_search"
hadoop_queue="bdp_jmart_ad.jd_ad_search"
# 3. 如果不支持bdpclient需要添加以下配置
#hadoop_user_certificate="your certificate"


#可选，训练数据本地路径，默认为./data, 动态修改, 运行于Cloud上时会动态修改为Cloud端路径
local_data_dir="./data"
log_dir="/export/App/training_platform/PinoModel/models"
#可选，可自定义入口文件，默认为main.py
entrance_filename="run.py"
#################################
# 命令行参数，单机的话参数都写在这里就好了
global_parameters = dict(
    dataset = 'amazon-book',
    regs = '[1e-4]',
    embed_size = 64,
    layer_size = '[64,64]',
    lr = 0.0001,
    batch_size = 2048//6,
    epoch = 1000
)
#################################
roles=['ps', 'worker']
# 必选，对应role的资源配置
ps = dict(
    count=1, # role数量
    cpu=16.0, # 单位为核
    mem=100.0, # 单位为G
    disk=16.0,   # 单位为G
    framework="9n-train:tf_1.12.0_cpu_ads_9ncloud_ana3_jupyter_py2_nm_bdpclient",
#    framework="9n-train:tf_1.12.0_cpu_ads_9ncloud_ana3_jupyter_py3_nm_bdpclient",
    gpu=0,
    job_name='ps'
)
worker = dict(
    count=6,
    cpu=20.0, # 单位为核
    mem=220.0, # 单位为G
    disk=100.0,   # 单位为G
    framework="9n-train:tf_1.12.0_gpu_ads_9ncloud_cuda9_cudnn7_py2_nm_bdpclient",
#    framework="9n-train:tf_1.12.0_gpu_ads_9ncloud_cuda9.0_cudnn7_nm_bdpclient",
    gpu=4,  #单位为卡
    job_name='worker'
)
#可选，若相同roles需不同参数，在此添加，下边分布式的例子会体现
worker_0 = dict(
    task_index=0
)
worker_1 = dict(
    task_index=1
)
worker_2 = dict(
    task_index=2
)
worker_3 = dict(
    task_index=3
)
worker_4 = dict(
    task_index=4
)
worker_5 = dict(
    task_index=5
)
worker_6 = dict(
    task_index=6
)
