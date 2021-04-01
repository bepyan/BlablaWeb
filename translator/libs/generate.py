import os
import pickle
import torch

from torch.autograd import Variable

def sample_model(opt, gru, prime_text=" ", length=1000, text=True): 
    # Make prime tensor
    text = [opt.vocab_ctoi.get(c, 0) for c in prime_text]
    prime_input = Variable(torch.LongTensor(text).view(-1, 1)).cuda()

    h = gru.init_hidden()
    output, h = gru(prime_input, h)
    
    for i in range(length+10):
        next_word = output[-1,:,:].squeeze().max(dim=0)[1].item() # word_discriminator
        if i > length and opt.vocab_itoc[next_word] == ' ':
            break
        prime_text += opt.vocab_itoc[next_word]
        
        # Make input tensor using the last word generated
        char_input = Variable(torch.LongTensor([next_word]).view(-1, 1)).cuda()
        
        output, h = gru(char_input, h)
        h = h.detach()
    return prime_text, h


def generate(_opt, _gru, str, len=1000):
    result, h = sample_model(opt=_opt,
                            gru=_gru,
                            prime_text=str,
                            length=len)
    return result

if __name__ == "__main__":
    opt=opt()
    model = gru(opt).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.dir, "gru.pkl")))
    
    result, h = sample_model(opt=opt,
                            gru=model,
                            prime_text="예문을 보면",
                            length=1000)
    print(result)