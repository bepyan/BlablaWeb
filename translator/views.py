from django.shortcuts import render
from django.http import HttpResponse

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


# Create your views here.
def translateRequest(request):
    return render(request, 'translator/translate.html')

def translateResponse(request):
    from translator.data import MODEL_ORIGINAL, MODEL_TARGET
    from translator.libs import BlahBlah
    import json

    result = BlahBlah.predictSentence(MODEL_ORIGINAL, MODEL_TARGET, request.POST['input'])
    return HttpResponse(json.dumps(result, ensure_ascii = False))

def completeSentence(request):
    return HttpResponse(request.POST['input'] + '</br>' + make_sentence(kogp2, vocab, tok, request.POST['input'][-500:]))
    
def nextSentence(request): 
    return HttpResponse(request.POST['input'] + '</br>' + make_sentence(kogp2, vocab, tok, generate(opt, gru, request.POST['input'][-500:], len=7)))