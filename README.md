# chatbot
...
- - - 
...
#충남 챗봇 수업의 저장소입니다.


- - -
...
## 2023_9_4

* 나의 첫 깃허브 메세지 입니다.

* 파이토치 프로그래밍 시작!

* 공유에 필요한 정보
    * freshmea

## 2023_9_20 

* arima모델을 사용하여 자전거 매출을 그래프로 나타냄

* arima_model을 arima.model로 바꿈

* "TypeError: ARIMA.fit() got an unexpected keyword argument 'disp'" 이런 에러가 떠서 13번줄에   'disp=0을 지움

* LSTM구조
    * LSTM순저파
        * 망각 게이트
            * 0이면 이전정보는 버리고 1이면 보존한다

        * 입력 게이트
            * 현재 정보 보존량을 결정

        * 셀
            * 은닉 노드를 메모리 셀이라고한다


## 2023_9_22

* LSTM모델의 학습및 성능을 확인함 정확도가 97%로 상당히 높다

* LSTM 계층을 스타벅스 주가 데이터셋으로 구현함

* 날짜 칼럼을 인덱스로 사용하는 과정에서 에러가났다['data']를 ['Data']로 변경함


## 2023_9_25

* 조기 종료를 이용한 성능 최적화

* 자연어 처리
    * 토큰화

        * 토큰화: 주어진 텍스트를 단어/문자 단위로 자르는 것
                  문장 토큰화와 단어 토큰화로 구분된다

* 전처리
    * 한글 토큰화
        * KoNLPy 라이브러리 사용함

    * 불용어 제거
        * 불용어: 문장 내에서 빈번하게 발생하여 의미를 부여하기 어려운 단어

    * 어간 추출
        * 어간 추출과 표제어 추출은 단어 원형을 찾아 주는 것이다.

    * 정규화
        * 정규화: 데이터셋이 가진 특성의 모든 데이터가 동일한 정도의 범위를 갖도록하는것


## 2023_9_26

* 임베딩
    * 임베딩:사람이 사용하는 언어를 컴퓨터가 이해할 수 있는 언어 형태인 벡터로 변환한 결과 혹은 일련의 과정을 의미함

    * 임베딩은 방법에 따라 '희소 표현 기반 임베딩', '횟수 기반 임베딩', '예측 기반 임베딩', '횟수/예측 기반 임베딩으로 나뉘어 진다.

* 트랜스포머 어텐션
    * 어텐션: 주로 언어 번역에서 사용되기 때문에 인코더와 디코더 네트워크를 사용한다.

    * 입력에 대한 벡터 변환을 인코더에서 처리하고 모든 벡터를 디코더로 보낸다.

    * 장점: 모든 벡터를 전달하여 시간이 흐를 수록 초기 정보를 잃어버리는 기울기 소멸 문제 해결

    * 단점: 모든 벡터가 전달되기 때문에 행렬 크기가 굉장히 커진다

* seq2seq
    * seq2seq: 입력 시퀀스에 대한 출력 시퀀스를 만들기 위한 모델이다

## 2023_10_11

* kaggle 경진대회 도전 타이타닉 생존자 예측
