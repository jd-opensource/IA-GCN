if [ ! -d ~/.pip ];then
    mkdir ~/.pip
    cp pip.conf ~/.pip

    mkdir tools
    pip install --target=tools --upgrade torch torchvision
fi

#nohup python -u Light_GCN_zhujian.py --dataset='gowalla' --regs='[1e-4]' --embed_size=64 --layer_size='[64,64,64]' --lr=0.001 --batch_size=256 --epoch=1000 2>&1 > nohup.out &
#tailf nohup.out
