import torch
import os

"""
# 백그라운드에 돌릴 것.
opt = opt()
gru = gru(opt).cuda()
gru.load_state_dict(torch.load(os.path.join(opt.dir, "gru.pkl")))
kogp2, vocab, tok = get_pytorch_kogpt2_model()
"""

def make_sentence(kogp2, vocab, tok, sent=""):
  ori = sent
  toked = tok(sent)
  for i in range(30): 
    input_ids = torch.tensor([vocab[vocab.bos_token],]  + vocab[toked]).unsqueeze(0)
    pred = kogp2(input_ids)[0]
    gen = vocab.to_tokens(torch.argmax(pred, axis=-1).squeeze().tolist())[-1]
    if gen == '</s>':
        break
    sent += gen.replace('▁', ' ')
    toked = tok(sent)
  
  if sent[-1] not in {'.','!','?',"'",'"', ' '}:
    sent += '.'
  return sent[len(ori):]

"""
if __name__ == "__main__":
  opt = opt()
  gru = gru(opt).cuda()
  gru.load_state_dict(torch.load(os.path.join(opt.dir, "gru.pkl")))
  kogp2, vocab, tok = get_pytorch_kogpt2_model()
  
  print('|  이어나갈 글을 입력하세요  |')
  str = make_sentence(kogp2, vocab, tok, input())
  for i in range(1, 11):
    print(str)
    str = make_sentence(kogp2, vocab, tok, generate(opt, gru, str, len=7))
"""

"""
1. make_sentence로 사용자 문장을 완성한다.
2. 우선권 수식어를 붙인다.
3. 5문장 정도를 이어 작문한다. (추가 작문 버튼을 누르면 계속 작문)
3' 작문한 문장이 맘에 드면 여기에도 수식어를 붙일 수 있도록 버튼

2-1 수식어를 선택한다. 작문 버튼 누르면 이에 따라 다시 작문
"""