The base models were implemented in a previous Github repository. We followed the instructions provided in that repository to implement them. 

Create a virtual environment and install the following packages:
(they can be placed in a requirements file)

conda create -n mctrans pip python=3.9

conda activate mctrans
git clone https://github.com/JiYuanFeng/MCTrans.git
cd MCTrans
python setup.py develop
pip install -r requirements.txt

--------------------------------------------------------------

absl-py==0.13.0
addict==2.4.0
cachetools==4.2.2
certifi==2021.5.30
chardet==4.0.0
future==0.18.2
google-auth==1.32.1
google-auth-oauthlib==0.4.4
grpcio==1.38.1
idna==2.10
importlib-metadata==4.6.1
joblib==1.0.1
Markdown==3.3.4
mmcv==1.3.9
monai==0.5.3
numpy==1.21.0
oauthlib==3.1.1
opencv-python==4.5.3.56
Pillow==8.3.1
prettytable==2.1.0
protobuf==3.17.3
pyasn1==0.4.8
pyasn1-modules==0.2.8
PyYAML==5.4.1
requests==2.25.1
requests-oauthlib==1.3.0
rsa==4.7.2
scikit-learn==0.24.2
scipy==1.7.0
six==1.16.0
sklearn==0.0
tensorboard==2.5.0
tensorboard-data-server==0.6.1
tensorboard-plugin-wit==1.8.0
threadpoolctl==2.2.0
typing-extensions==3.10.0.0
urllib3==1.26.6
wcwidth==0.2.5
Werkzeug==2.0.1
yapf==0.31.0
zipp==3.5.0

After installing these packages, install the following PyTorch packages

conda install pytorch=1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install numpy==1.21.0
conda install Pillow==8.3.1
conda install six==1.16.0
conda install typing-extensions==3.10.0.0

-----------------------------------------------------------------------------

Compile other CUDA operators (MultiScaleDeformableAttention)https://github.com/fundamentalvision/Deformable-DETR

cd mctrans/models/ops/
bash make.sh


------------------------------------------------------------------------------
after installation go to

MCTrans/configs/_base_/datasets/pannuke.py

to change the workers and groups to a
number that the computer accepts, also the patch_size, adjust it to the size
of the input images.

-------------------------------------------------

then adjust to configure the model, epochs, etc.
We have at the following address
MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 
The 4 files that define the model, the number of epochs, among other parameters.

To modify the number of epochs we will go to:
/home/graciela/MCTrans/configs/_base_/schedules/pannuke_bs32_ep400.py

And we adjust the "max_epochs" variable, you can also adjust the learning rate, interval
at which the models are saved.
------------------------------------------------------------------
At the address:

/home/graciela/MCTrans/configs/_base_/models/

We have the 14 models implemented in the github. Here we only need to copy
the name of the file we want to use, and paste it into the next file:

MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 

In the part corresponding to the model.
-----------------------------------------------------------------
We will give an example.

In the following file:
MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 

We have the following:

_base_ = [
    '../_base_/models/unet++_res34_d5.py', '../_base_/datasets/pannuke.py',
    '../_base_/default_runtime.py', '../_base_/schedules/pannuke_bs32_ep400.py'
]

The model being used is the unet++_res34_d5.py, to change it for example
to the mctrans model that uses vgg, we just have to go to:

/home/graciela/MCTrans/configs/_base_/models/

Copy the name of the file that has the mctrans vgg, which is: mctrans_vgg32_d5.py
And paste that name into the file:

MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 

When pasting it we will have the following:

_base_ = [
    '../_base_/models/mctrans_vgg32_d5.py', '../_base_/datasets/pannuke.py',
    '../_base_/default_runtime.py', '../_base_/schedules/pannuke_bs32_ep400.py'
]


---------------------------------------------------------------
To run an experiment, open spyder, and open the file:
/home/graciela/MCTrans/tools/train.py

--config "/home/graciela/MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py" --work-dir "/home/graciela/MCTrans/work_dirs/" --load-from "/home/graciela/MCTrans/work_dirs/best_mDice_epoch_688.pth"

At the top, in the "Run" tab, select the "Configuration by file" option,
in the popup window, in the "General Settings" section
select "Command line options:" and enter:
--config "/home/graciela/MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py" --work-dir "/home/graciela/MCTrans/work_dirs/"
And run.

NOTE: It depends on spyder, but if the "Command line options" box does not appear,
make the execution configuration window larger and the box should appear.
