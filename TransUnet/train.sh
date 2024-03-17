
cd /home/graciela/Adrian/TransUnet/transunet_pytorch/
conda activate unetpytorch

python main.py --mode train --model_path /home/graciela/Adrian/TransUnet/transunet_pytorch/model/model1.pth --train_path /home/graciela/Adrian/TransUnet/transunet_pytorch/trainset --test_path /home/graciela/Adrian/TransUnet/transunet_pytorch/testset
