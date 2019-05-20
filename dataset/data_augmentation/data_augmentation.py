

from konlpy.tag import Twitter
twitter = Twitter()
print(twitter.morphs(u'단독입찰보다 복수입찰의 경우'))
# ['단독', '입찰', '보다', '복수', '입찰', '의', '경우', '가']
print(twitter.nouns(u'유일하게 항공기 체계 종합개발 경험을 갖고 있는 KAI는'))
# ['유일하', '항공기', '체계', '종합', '개발', '경험']
print(twitter.phrases(u'날카로운 분석과 신뢰감 있는 진행으로'))
# ['분석', '분석과 신뢰감', '신뢰감', '분석과 신뢰감 있는 진행', '신뢰감 있는 진행', '진행', '신뢰']
print(twitter.pos(u'이것도 되나욬ㅋㅋ'))
# [('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되나욬', 'Noun'), ('ㅋㅋ', 'KoreanParticle')]
print(twitter.pos(u'이것도 되나욬ㅋㅋ', norm=True))
# [('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되', 'Verb'), ('나요', 'Eomi'), ('ㅋㅋ', 'KoreanParticle')]
print(twitter.pos(u'이것도 되나욬ㅋㅋ', norm=True, stem=True))
# [('이', 'Determiner'), ('것', 'Noun'), ('도', 'Josa'), ('되다', 'Verb'), ('ㅋㅋ', 'KoreanParticle')]

print([ noun for (noun, postag) in twitter.pos(u'이것도 되나욬ㅋㅋ', norm=True, stem=True) if postag == 'Noun'])

u'이것도 되나욬ㅋㅋ'


print([ noun for (noun, postag) in twitter.pos(u'저만 안되는 건가요?', norm=True, stem=True) if postag == 'Noun'])
print([ noun for (noun, postag) in twitter.pos(u'&lt;게임 내 오류나 불편하신 사항을 경험하셨나요?&gt;1.사용중인 캐릭터명/캐릭터 정보를 알려주세요 - II봄날II Ex) GM강철, 휴먼-나이트메어 각성2.발생일시 - 2019.01.30 오후 6시 30분 Ex) 2018.01.01 오후 9시 30분 경3.사용중인 기기명과 OS - 갤럭시 S9+ Ex) 갤럭시 S8, 안드로이드3.발생현상/재현 가능한 상황 - 경험치 300% 미적용 Ex) 게임 접속 시 메인로비로 이동이 안되며 검은 화면만 나옵니다.4.관련 스크린 샷 또는 동영상 -', norm=True, stem=True) if postag == 'Noun'])