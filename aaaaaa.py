chk3 = 'y'
while chk3 == 'y' : 
    chk = chk2 = 1
    while (chk == chk2 == 0) == False : #금액과 할부 기간이 정상입력 될 때까지 반복
        if chk : 
            a = input("금액을 입력하세요 : ") #금액 입력
            if (len(a) == 0) : continue #Enter 입력으로 인한 오류 방지
            chk = 0
            for i in range(0, len(a) + 1) :
                if i == len(a) : p = float(a); break #정상 입력 확인 후, 금액 저장
                if a[i] > '9' or a[i] < '0' : chk = 1; print("input error"); break #입력값에 문자 포함 시, 에러 출력
        else :
            a = input("할부 기간을 입력하세요(금액을 다시 입력하시려면 back을 입력해주세요. 현재 금액 : %d) : " %p) #할부 기간 입력
            if not a : continue #Enter 입력으로 인한 오류 방지
            if (a == "back") : chk = 1; continue #금액 입력 구간으로 이동
            chk2 = 0
            for i in range(0, len(a)) :
                if a[i] > '9' or a[i] < '0' : chk2 = 1; print("input error"); break #입력값에 문자 포함 시, 에러 출력
    print("매달 납부할 금액은 %.0f입니다." %round(p / float(a))) #매달 남부할 금액 출력
    chk3 = input("다시 계산하시겠습니까?(y/n) : ")
print("End of process") # 종료