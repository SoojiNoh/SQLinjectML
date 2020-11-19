
Anacoda (Windows) 다운받고, yml 파일로부터 conda 환경 복사받으셈

**conda env create -f conda_SQLinjectML_env.yml#**


conda 실행시키셈
**conda activate tf_env**

log_classification 프로젝트 Eclipse로 열고
Eclipse MarketPlace에서 PyDev 플러그인 다운

플러그인 PyDev 패키지 익스플로러 창 내 log_classification 프로젝트 오른쪽 마우스
\>> 'properties' >> 'PyDev-Interprete'r >>'Click Here to Configure an interpreter not listed' >>' open interpreter preference page' >> 'Browse for python/pypy package'
>> ~\anaconda3\envs\tf_env\python.exe 선택 후 'OK'
>> 'Apply and Close'
>> 'Interpreter' 선택 중 'tf_env' 선택. >> 'Apply and Close'
