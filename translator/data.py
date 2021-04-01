from translator.libs import BlahBlah
#dir = 'C:/Users/reldn/BLABLA/BlablaWeb/translator/libs/models/'
dir = 'C:/Users/XPS/Desktop/SIG/BlablaWeb/translator/libs/models/'
MODEL_ORIGINAL = BlahBlah.loadModel(dir + 'original.model')
MODEL_TARGET = BlahBlah.loadModel(dir + 'target.model')

import torch, os
from translator.libs.generate import generate
from translator.libs.write import make_sentence
from translator.libs.models.util import get_pytorch_kogpt2_model
from translator.libs.models.gru import gru
from translator.libs.models.opt import opt

kogp2, vocab, tok = get_pytorch_kogpt2_model()
opt = opt()
gru = gru(opt).cuda()
gru.load_state_dict(torch.load(os.path.join(opt.dir, "gru.pkl")))