Instalar MCTrans

Crear un ambiente virtual e instalar los paquetes siguientes paquetes:
(pueden colocarse en un archivo requirements)


conda create -n mctrans pip python=3.7
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

Despues de instalar estos paquetes, instalar los siguientes paquetes
de pytorch

conda install pytorch=1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
conda install numpy==1.21.0
conda install Pillow==8.3.1
conda install six==1.16.0
conda install typing-extensions==3.10.0.0

-----------------------------------------------------------------------------

Compilar otros operadores de CUDA (MultiScaleDeformableAttention)https://github.com/fundamentalvision/Deformable-DETR

cd mctrans/models/ops/
bash make.sh


------------------------------------------------------------------------------
despues de instalar ir a 

MCTrans/configs/_base_/datasets/pannuke.py

para cambiar los workers y grupos a un
numero que acepte la computadora, tambien el patch_size, ajustarlo al tamaño
de las imagenes de entrada.

-------------------------------------------------

despues ajustar para configurar el modelo, epocas,etc. 
Tenemos en la siguiente direccion 
MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 
Los 4 archivos que definen el modelo, el numero de epocas, entre otros parametros.

Para modificar el numero de epocas vamos a:
/home/graciela/MCTrans/configs/_base_/schedules/pannuke_bs32_ep400.py

Y ajustamos la variable "max_epochs", tambien puede ajustarse el learning rate, intervalor
en que se guardan los modelos.

------------------------------------------------------------------
En la direccion:

/home/graciela/MCTrans/configs/_base_/models/

Tenemos los 14 modelos implementados en el github. Aqui unicamente tenemos que copiar
el nombre del archivo que queramos usar, y pegarlo en el archivo siguiente:

MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 

En la parte que corresponde al modelo. 
-----------------------------------------------------------------
Daremos un ejemplo.

En el archivo siguiente:
MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 

Tenemos lo siguiente:

_base_ = [
    '../_base_/models/unet++_res34_d5.py', '../_base_/datasets/pannuke.py',
    '../_base_/default_runtime.py', '../_base_/schedules/pannuke_bs32_ep400.py'
]

El modelo que se esta usando es el unet++_res34_d5.py, para cambiarlo por ejemplo
al modelo mctrans que utilice la vgg, tenemos que unicamente ir a:

/home/graciela/MCTrans/configs/_base_/models/

Copiar el nombre del archivo que tiene la mctrans vgg, que es: mctrans_vgg32_d5.py
Y pegar ese nombre en el archivo:

MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py 

Al pegarlo tendremos lo siguiente:

_base_ = [
    '../_base_/models/mctrans_vgg32_d5.py', '../_base_/datasets/pannuke.py',
    '../_base_/default_runtime.py', '../_base_/schedules/pannuke_bs32_ep400.py'
]


---------------------------------------------------------------
Para correr un experimento, abrir spyder, y abrir el archivo:
/home/graciela/MCTrans/tools/train.py

--config "/home/graciela/MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py" --work-dir "/home/graciela/MCTrans/work_dirs/" --load-from "/home/graciela/MCTrans/work_dirs/best_mDice_epoch_688.pth"

En la parte superior, en la pestaña "Ejecutar", seleccionar la opcion "configuracion por archivo",
en la ventana emergente, en la seccion "Ajustes generales"
seleccionar "Opciones de linea de comandos:"  y colocar: 
--config "/home/graciela/MCTrans/configs/pannuke-vgg32/mctrans_vgg32_d5_256x256_400ep_pannuke.py" --work-dir "/home/graciela/MCTrans/work_dirs/"
Y ejecutar.

NOTA: Depende el spyder, pero si en "Opciones de linea de comandos no sale cuadro para poner texto,
hacer mas grande la ventana de configuracion de ejecucion por archivo y debe salir el cuadro.