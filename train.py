from lpnet import lpnet
from load_data import *

from torch.utils.data import *

numPoints = 4
numClasses = 36
batchSize = 32
input_shape = (480, 480)
model_name = 'base'
img_dir = 'img'
epochs = 25
num_workers = 4

ads = {'0' : '_', '1' : 'A', '2' : 'B', '3' : 'C', '4' : 'D', '5' : 'E',
       '6' : 'F', '7' : 'G', '8' : 'H', '9' : 'I', '10' : 'J',
       '11' : 'K', '12' : 'L', '13' : 'M', '14' : 'N', '15' : 'O',
       '16' : 'P', '17' : 'Q', '18' : 'R', '19' : 'S', '20' : 'T',
       '21' : 'U', '22' : 'V', '23' : 'W', '24' : 'X', '25' : 'Y',
       '26' : 'Z', '27' : '0', '28' : '1', '29' : '2', '30' : '3',
       '31' : '4', '32' : '5', '33' : '6', '34' : '7', '35' : '8',
       '36' : '9'}

provNum = {j : i for i, j in ads.items()}

lpnet = lpnet(numPoints, numClasses, model_name = model_name, net_file = None, base_file = None)
dst = lpDataLoader(img_dir, input_shape, provNum, model_name = model_name)
trainloader = DataLoader(dst, batch_size=batchSize, shuffle=True, num_workers= num_workers)
lpnet.train(trainloader, batchSize, epoch_start= 0, epochs= epochs)
