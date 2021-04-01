# -*- conding: utf-8 -*-
import os
import urllib3
import json
from gensim.models import Word2Vec

###
# API를 통한 언어 분석
###
def request(text):
    openApiURL = "http://aiopen.etri.re.kr:8000/WiseNLU"
    accessKey = "3832f203-84be-4742-9756-d9503d34b345"
    analysisCode = "srl"

    requestJson = {
        "access_key": accessKey,
        "argument": {
            "text": text,
            "analysis_code": analysisCode
        }
    }

    http = urllib3.PoolManager()
    response = http.request(
    "POST",
    openApiURL,
    headers={"Content-Type": "application/json; charset=UTF-8"},
    body=json.dumps(requestJson)
    )

    # print("[responseCode] " + str(response.status))
    # print("[responBody]")
    # print(str(response.data,"utf-8"))

    res = json.loads(response.data)
    if 'return_object' not in res:
        print(res)
        return None

    return res

###
# 의미역 분석을 통해 수식어 원형, 수식어 원본, 의미역 대상을 추출
# (학습을 통해 적절한 수식어 예측에 사용)
# (학습을 통해 원형 변환 예측에 사용)
###
def getSRL(res):
    SRL = []
    for sentence in res['return_object']['sentence']: # 문장 단위
        for srl in sentence['SRL']: # 의미역 단위
            srlWord = sentence['word'][int(srl['word_id'])] # 의미역의 단어
            srlType = ''

            checkFlag = False
            for check in ['.','?','!',',',':','/','"','(',')','-','_','~','*']: # 특수문자 포함 수식어 제거
                if check in srl['verb']: checkFlag = True
                if check in srlWord['text']: checkFlag = True
            if checkFlag == True:
                continue        

            for morp in sentence['morp']: # 해당 의미역의 품사 찾기
                if morp['lemma'] in srl['verb']:
                    srlType = morp['type']
                    break
            if srlType != 'VV':
                for arg in srl['argument']: # 각 arg를 탐색
                    argWord = sentence['word'][int(arg['word_id'])] # arg의 단어
                    argMorp = sentence['morp'][int(argWord['begin'])] # arg의 형태소
                    if (arg['type'] == 'ARG1') and (len(srl['verb']) > 1): # 대상이 ARG1(피동주), 수식어가 2글자 이상
                        pair = [srl['verb'] + '/V', srlWord['text']  + '/O', argMorp['lemma']  + '/N'] 
                        SRL.append(pair) # 추가
    return SRL

###
# 의미역 분석을 통해 의미역 대상을 추출
###
def getNoun(res):
    nouns = []

    for sentence in res['return_object']['sentence']: # 문장 단위
        for srl in sentence['SRL']: # 의미역 단위
            for arg in srl['argument']: # 각 ARG 탐색
                argMods = sentence['dependency'][int(arg['word_id'])]['mod'] # arg의 수식어
                if not argMods: # 수식어가 없는 경우만 추가
                    argWord = sentence['word'][int(arg['word_id'])] # arg의 단어
                    argMorp = sentence['morp'][int(argWord['begin'])] # arg의 형태소
                    if ('M' not in arg['type']) and (argMorp['type'] != 'NP'): # NP (대명사) 제외
                        nouns.append(argMorp['lemma']) # [명사의 원형, 단어의 위치] (** 현재: 원형만 추가)

    return nouns

###
# 저장한 데이터셋 파일을 다시 불러옴
###
def loadDataset(inputFile):
    fr = open(inputFile, 'r', encoding='UTF-8')
    result = []

    while True:
        # 라인 읽기 
        sentence = fr.readline()
        if not sentence: break
        sentence = sentence.replace('\n','')

        data = sentence.split(',')
        result.append([data[0], data[1]])

    fr.close()
    return result

###
# getSRL을 통해 생성한 데이터를 분리하여 데이터셋 파일로 저장
# [(수식어 원형, 수식어 원본, 의미역 대상)] > [(수식어 원형, 수식어 원본)] + [(의미역 대상, 수식어 원형)]
###
def saveDataset(SRLs, outputOriginal='original.csv', outputTarget='target.csv', mode='a'):
    fileOriginal = open(outputOriginal, mode, encoding='UTF-8')
    fileTarget = open(outputTarget, mode, encoding='UTF-8')

    for srl in SRLs: # SRL 단위
        fileOriginal.write(srl[0] + ',' + srl[1] + '\n')
        fileTarget.write(srl[2] + ',' + srl[0] + '\n')

###
# 추출한 데이터를 통해 W2V 학습
# [(A, B),(C, D)] 쌍의 데이터셋 필요
###
def saveModel(data, save='output.model'):

    # 모델 생성
    model = Word2Vec(data, sg=1, min_count=5, size=100, window=2, workers=4) 

    if save != None:
        model.save(save)

    return model


def loadModel(filePath):
    return Word2Vec.load(filePath)

###
# 명사에 적절한 수식어 예측
###
def predict(originalModel, targetModel, noun):
    result = []

    targetResult = targetModel.predict_output_word([noun + '/N']) # 수식어 예측 결과
    if targetResult != None:
        for targetWord in targetResult[:5]:
            # if targetWord[1] < 0.001: break # 너무 신뢰도가 낮은 데이터는 제외
            if '/V' not in targetWord[0]: continue # 정상 수식어 아니면 제외
            target = [str(targetWord[0]), int(targetWord[1]*1000)/10]
            originalResult = originalModel.predict_output_word([targetWord[0]])
            original = []
            if originalResult != None:
                for originalWord in originalResult[:5]:
                    # if originalWord[1] < 0.001: break # 너무 신뢰도가 낮은 데이터는 제외
                    if '/O' not in originalWord[0]: continue # 정상 수식어 원본 아니면 제외
                    original.append([str(originalWord[0].split('/')[0]),int(originalWord[1]*1000)/10])
            if original:
                target.append(original)
                result.append(target)
    return result

def predictSentence(originalModel, targetModel, sentence):
    res = request(sentence)
    if res == None:
        return None

    result = []
    nouns = getNoun(res)
    for noun in nouns:
        predictData = predict(originalModel, targetModel, noun)
        result.append([noun,predictData])

    return result

## ---------------------------------------------------------------------------
## 대용량 처리를 위한 함수

###
# 1) 디렉토리 내 파일을 모두 읽고, 1만자 단위로 새로 파일을 만들어 저장
# dataDir 내 original, cut 폴더 존재할 것
###
def convertTo1(dataDir):
    print("1) 문장 데이터 파일 > 1만자 단위 데이터 파일")

    sentence = ''
    sentences = ''
    count = 1

    # 각 파일 읽기
    for fname in os.listdir(dataDir + '/original'):
        fr = open(dataDir + '/original/' + fname, 'r', encoding='UTF-8')
        print(' 읽기 시작 (' + fname + ')')
        while True:
            # 라인 읽기 
            sentence = fr.readline()
            if not sentence: break
            sentence = sentence.replace('\n',' ')

            # 10000자 이상이면 저장
            if (len(sentences) + len(sentence) >= 10000):
                print('  쓰기 시작 (' + str(count) + ') - 글자수 : ' + str(len(sentences)) + '자')
                fw = open(dataDir + '/cut/' + str(count) + '.txt', 'w', encoding='UTF-8')
                fw.write(sentences)
                fw.close()
                sentences = ''
                count = count + 1
            
            # 10000자 이하이면 문장을 잇는다.
            sentences = sentences + sentence
        fr.close()

###
# 2) convertTo1()을 통해 얻은 1만자 단위의 파일들을 읽고 API에 결과 요청
# dataDir 내 cut, json 폴더 존재할 것
###
def convertTo2(dataDir):
    print("2) 1만자 단위 데이터 파일 > json 결과 파일")

    for fname in os.listdir(dataDir + '/cut'):
        print(" 파일: " + fname)
        fr = open(dataDir + '/cut/' + fname, 'r', encoding='UTF-8')
        fw = open(dataDir + '/json/' + fname, 'w', encoding='UTF-8')
        sentence = fr.readline()
        responseJson = request(sentence)
        fw.write(json.dumps(responseJson))

###
# 3) convertTo2()을 통해 얻은 json 파일들을 읽고 추출 결과로 변환
# dataDir 내 json, srl 폴더 존재할 것
###
def convertTo3(dataDir):
    print("3) json 결과 파일 > 수식어 추출 파일")

    for fname in os.listdir(dataDir + '/json'):
        print("파일: " + fname)
        fr = open(dataDir + '/json/' + fname, 'r', encoding='UTF-8')
        sentence = fr.readline()
        srl = getSRL(json.loads(sentence))
        saveDataset(srl, dataDir + '/srl/original/' + fname, dataDir + '/srl/target/' + fname, 'w')

###
# 4) convertTo3()을 통해 얻은 추출 파일들을 읽고 학습
# dataDir 내 srl 폴더 존재할것, srl 내 original, target, model 폴더 존재할 것
###
def convertTo4(dataDir):
    print("4) 수식어 추출 파일 > 모델 파일")

    original = []
    target = []

    for fname in os.listdir(dataDir + '/srl/original'):
        print("[original] 파일: " + fname)
        data = loadDataset(dataDir + '/srl/original/' + fname)
        original.extend(data)
    saveModel(original, dataDir + '/srl/model/original.model')

    for fname in os.listdir(dataDir + '/srl/target'):
        print("[target] 파일: " + fname)
        data = loadDataset(dataDir + '/srl/target/' + fname)
        target.extend(data)
    saveModel(target, dataDir + '/srl/model/target.model')
    
    

###
# 테스트 코드
###

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore") # 경고 무시
    print("------------------------------------------------------------------")
    MENU = 4

    if MENU == 1: # 파일 쪼개기
        print('1) 경로 입력 (형식 : C:\\...\\...\\dataset)')
        dataDir = input()
        convertTo1(dataDir)

    elif MENU == 2: # 데이터 요청
        print("2) 경로 입력 (형식 : C:\\...\\...\\dataset)")
        dataDir = input()
        convertTo2(dataDir)

    elif MENU == 3: # 데이터셋 생성
        print("3) 경로 입력 (형식 : C:\\...\\...\\dataset)")
        dataDir = input()
        convertTo3(dataDir)

    elif MENU == 4: # 모델 생성
        print("4) 경로 입력 (형식 : C:\\...\\...\\dataset)")
        dataDir = input()
        convertTo4(dataDir)

    elif MENU == 5: # 테스트1
        print("5) 경로 입력 (형식 : C:\\...\\...\\dataset)")
        dataDir = input()
        originalModel = Word2Vec.load(dataDir + '/srl/model/original.model')
        targetModel = Word2Vec.load(dataDir + '/srl/model/target.model')
        while True:
            print(">> 명사를 입력하세요.")

            data = predict(originalModel, targetModel, input())
            print(data)

    elif MENU == 6: # 테스트2
        dataDir = 'BlablaWeb/translator/libs/models/'
        originalModel = Word2Vec.load(dataDir + 'original.model')
        targetModel = Word2Vec.load(dataDir + 'target.model')
        while True:
            print(">> 문장을 입력하세요.")

            result = predictSentence(originalModel, targetModel, input())
            print(result)

    print("------------------------------------------------------------------")