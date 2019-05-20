import random
from copy import deepcopy
try:
    from dataset.data_augmentation.typo_generator import Hangulpy as hp
except:
    from dataset.data_augmentation import Hangulpy as hp

def make_noise(text, num=1):
    result=[]
    text = deepcopy(text)
    for n in range(num):
        result.append(korean_noise(text))
            
    return result

confusion_dic = {
    'ㄱ' : ['r','R','ㄲ','ㅅ','ㄷ','ㄹ'],
    'ㄲ' : ['ㄱ','R'],
    'ㅂ' : ['q','Q','ㅃ','ㅈ','ㅁ'],
    'ㅃ' : ['ㅂ','Q'],
    'ㅈ' : ['w','W','ㅉ','ㄷ','ㅂ','ㄴ'],
    'ㅉ' : ['ㅈ','W'],
    'ㄷ' : ['e','E','ㄸ','ㅈ','ㄱ','ㅇ'],
    'ㄸ' : ['ㄷ','E'],
    'ㅅ' : ['t','T','ㅆ','ㄱ','ㅛ','ㅎ'],
    'ㅆ' : ['ㅅ','T'],
    'ㅛ' : ['y','Y','ㅕ','ㅗ','ㅅ'],
    'ㅕ' : ['u','U','ㅛ','ㅓ','ㅑ'],
    'ㅑ' : ['i','I','ㅕ','ㅐ','ㅏ'],
    'ㅐ' : ['o','O','ㅒ','ㅑ','ㅔ','ㅣ'],
    'ㅒ' : ['ㅐ','O'],
    'ㅔ' : ['p','P','ㅖ','[','ㅐ',';'],
    'ㅖ' : ['ㅔ','P'],
    'ㅁ' : ['a','A','ㄴ','ㅂ','ㅋ'],
    'ㄴ' : ['s','S','ㅁ','ㅇ','ㅈ','ㅌ'],
    'ㅇ' : ['d','D','ㄴ','ㄹ','ㄷ','ㅊ'],
    'ㄹ' : ['f','F','ㅇ','ㄱ','ㅎ','ㅍ'],
    'ㅎ' : ['g','G','ㄹ','ㅅ','ㅗ','ㅠ'],
    'ㅗ' : ['h','H','ㅎ','ㅛ','ㅓ','ㅜ'],
    'ㅓ' : ['j','J','ㅗ','ㅏ','ㅕ','ㅡ'],
    'ㅏ' : ['k','K','ㅓ','ㅣ','ㅑ',','],
    'ㅣ' : ['l','L','ㅏ',';','ㅐ','.'],
    'ㅋ' : ['z','Z','ㅁ','ㅌ'],
    'ㅌ' : ['x','X','ㅋ','ㄴ','ㅊ'],
    'ㅊ' : ['c','C','ㅌ','ㅍ','ㅇ'],
    'ㅍ' : ['v','V','ㅊ','ㅠ','ㄹ'],
    'ㅠ' : ['b','B','ㅍ','ㅜ','ㅎ'],
    'ㅜ' : ['n','N','ㅠ','ㅡ','ㅗ'],
    'ㅡ' : ['m','M','ㅜ',',','ㅓ']
}
    
def korean_noise(sent):
    try:
        num_try=0
        while True:
            num_try+=1

            if num_try>10:
                return sent

            c_index = random.choice(range(len(sent)))
            c = sent[c_index] 
            if hp.is_all_hangul(c)==False: continue
            if hp.has_jongsung(c):
                if random.uniform(0,1) > 0.5:
                    try:
                        decomposed = hp.decompose(c)
                        if random.uniform(0,1) > 0.6:
                            cho = decomposed[0]
                            joong = decomposed[1]
                        elif random.uniform(0,1) > 0.3:
                            cho = random.choice(confusion_dic[decomposed[0]])
                            joong = decomposed[1]
                        else:
                            cho = decomposed[0]
                            joong = random.choice(confusion_dic[decomposed[1]])
                        try:
                            composed = hp.compose(cho,joong)
                        except:
                            composed = "".join([cho,joong])

                    except:
                        continue

                    if random.uniform(0,1) > 0.5:
                        if random.uniform(0,1) > 0.5:
                            noised = composed + decomposed[-1]
                        else:
                            noised = composed + random.choice(confusion_dic[decomposed[-1]])
                    else:
                        noised = composed
                else:
                    try:
                        decomposed = hp.decompose(c)
                        if random.uniform(0,1) > 0.5:
                            composed = hp.compose(decomposed[2],decomposed[1],decomposed[0]) # 초성,중성 스위치
                        else:
                            composed = hp.compose(decomposed[2],decomposed[1]) 
                    except:
                        continue

                    noised = composed
            else:
                decomposed = hp.decompose(c)
                if random.uniform(0,1) > 0.6:
                    cho = decomposed[0]
                    joong = decomposed[1]
                elif random.uniform(0,1) > 0.3:
                    cho = random.choice(confusion_dic[decomposed[0]])
                    joong = decomposed[1]
                else:
                    cho = decomposed[0]
                    joong = random.choice(confusion_dic[decomposed[1]])
                try:
                    noised = hp.compose(cho,joong)
                except:
                    noised = "".join([cho,joong])
            break

        if c_index<len(sent)-1:
            sent = sent[:c_index] + noised + sent[c_index+1:]
        else:
            sent = sent[:c_index]+noised
    except:
        pass

    return sent

if __name__ == '__main__':
    print(make_noise(text = "정재우야 감사합니다. 왜이렇게 나오는거야?", num=5))
    print(make_noise("정재우야 감사합니다. 왜이렇게 나오는거야?"))
    print(make_noise("정재우야 감사합니다. 왜이렇게 나오는거야?"))
    print(make_noise("정재우야 감사합니다. 왜이렇게 나오는거야?"))
    print(make_noise("정재우야 감사합니다. 왜이렇게 나오는거야?"))