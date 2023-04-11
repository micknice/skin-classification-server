import json
import torch
from commons import get_net, get_img_tensor
import random
from numpy import random

net=get_net().to('cpu')


def get_skin_prediction(image_bytes):
    tensor = get_img_tensor(image_bytes)
    with torch.no_grad():
        output = net(tensor) 
        output = torch.nn.functional.softmax(output, dim=-1)        
        assessment = int(torch.argmax(output, dim=-1))        
        return assessment

