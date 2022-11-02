import os
import numpy as np
import pandas as pd
import multiprocessing


file_path = '/home/aipaas/mpower/21cn'
excel_file = 'result_summary.xlsx'

df = pd.read_excel(os.path.join(file_path, excel_file), sheet_name='audit')
num = len(df)
df1 = df[(df.human == 'pass')|(df.human == 'other')]
print("total:{}, pass:{}, illegal:{}".format(num, len(df1), (num - len(df1))))
porn_up = 0.9
politic_low = 0.2
terror_up = 0.9


def porn_terror_label(value, low_th, upp_th):
    if value != 'null':
        if value < low_th:
            return 0
        elif value > upp_th:
            return 1
        else:
            return 2
    elif value is None:
        return 0
    else:
        return 2


def politic_label(value, low_th, upp_th):
    if value != 'null':
        if value < low_th:
            return 1
        elif value > upp_th:
            return 0
        else:
            return 2
    elif value is None:
        return 0
    else:
        return 2


def result_statistic(result:list):
    true_pos = 0  # tp
    true_neg = 0  # tn
    false_pos = 0  # fp
    false_neg = 0  # fn
    pred_doubt = 0

    normal = ['pass', 'other']
    # infringement: 侵权
    illegal = ['advertise', 'gambling', 'infringement', 'pornography', 'privacy']
    tn_id = []
    tp_id = []

    for rst in result:
        value = rst[0] + rst[1] + rst[2] + rst[3]
        risk_label = rst[0:4]
        if rst[4] in normal:
            if value == 0:
                true_neg += 1
                tn_id.append(rst[5])
            elif value > 0:
                if 1 in risk_label:
                    false_pos += 1
                elif 2 in risk_label:
                    pred_doubt += 1
        elif rst[4] in illegal:
            if value == 0:
                false_neg += 1
            elif value > 0:
                if 1 in risk_label:
                    true_pos += 1
                    tp_id.append(rst[5])
                elif 2 in risk_label:
                    pred_doubt += 1
    return true_pos, true_neg, false_pos, false_neg, pred_doubt


# id	porn	politic	terror	ocr	human
def calculate_metrics(porn_th, politic_tl, politic_th, terror_th, index):
    flag = 0
    pid = os.getpid()
    all = porn_th.__len__() * politic_th.__len__() * terror_th.__len__()
    thresh = porn_th[0]

    log_dir = '/your/path/to/log'
    log_name = 'log_name.txt'
    log_path = os.path.join(log_dir, log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    with open(log_path, 'a+') as ff:
        print("pid,iter,porn,politic,terror,tp, tn, fp, fn, doubt,漏检率,误判率,验出率, 人工审核")
        ff.write("iter,porn,politic,terror,tp,tn,fp,fn,doubt,漏检率,误判率,验出率,人工审核\n")
        for i in porn_th:
            for j in politic_th:
                for pl in politic_tl:
                    for k in terror_th:
                        result = []
                        flag += 1
                        for d in range(num):
                            porn = porn_terror_label(df['porn'][d], i, porn_up)
                            politic = politic_label(df['politic'][d], pl, j)
                            terror = porn_terror_label(df['terror'][d], k, terror_up)
                            ocr = porn_terror_label(df['ocr'][d], 0.1, 0.9)
                            result.append((porn, politic, terror, ocr, df['human'][d], df['id'][d]))

                        tp,tn,fp,fn,doubt = result_statistic(result)
                        miss_tp = fn/num # (num - fn - tn - doubt)/num
                        false_alarm = fp/num
                        acc = (tp+tn)/num
                        re_audit = doubt/num
                        ff.write("{}/{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                                 format(flag,all, i,j,k,tp,tn,fp,fn,doubt, round(miss_tp, 12), round(false_alarm, 8), round(acc, 8), round(re_audit, 8)))
                        print("process:{},{}/{},{},{},{},{},{},{},{},{},{},{},{},{}".
                                 format(index, flag,all, i,j,k,tp,tn,fp,fn,doubt, round(miss_tp, 12), round(false_alarm, 8), round(acc, 8), round(re_audit, 8)))


def calculate_metrics2(porn_low, porn_high, politic_tl, politic_th, terror_th, index):
    flag = 0
    pid = os.getpid()
    all = politic_tl.__len__() * politic_th.__len__() * terror_th.__len__()
    log_dir = '/your/path/to/result'
    log_name = 'threshold_search_result.txt'
    log_path = os.path.join(log_dir, log_name)
    if os.path.exists(log_path):
        os.remove(log_path)

    with open(log_path,'a+') as ff:
        print("pid,iter,porn_low,porn_high,politic_low,politic_high,terror,tp, tn, fp, fn, doubt,漏检率,误判率,验出率, 人工审核")
        ff.write("iter,porn_low,porn_high,politic_low,politic_high,terror,tp,tn,fp,fn,doubt,漏检率,误判率,验出率,人工审核\n")
        for ph in politic_th:
            for pl in politic_tl:
                for k in terror_th:
                    result = []
                    flag += 1
                    for d in range(num):
                        porn = porn_terror_label(df['porn'][d], porn_low, porn_high)
                        politic = politic_label(df['politic'][d], pl, ph)
                        terror = porn_terror_label(df['terror'][d], k, terror_up)
                        ocr = porn_terror_label(df['ocr'][d], 0.1, 0.9)
                        result.append((porn, politic, terror, ocr, df['human'][d], df['id'][d]))

                    tp,tn,fp,fn,doubt = result_statistic(result)
                    miss_tp = fn/num # (num - fn -tn -doubt)/num which depends
                    false_alarm = fp/num
                    acc = (tp+tn)/num
                    re_audit = doubt/num
                    ff.write("index:{},{}/{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                             format(index,flag,all, porn_low, porn_high,ph,pl,k, tp,tn,fp,fn,doubt,
                                    round(miss_tp, 12), round(false_alarm, 8), round(acc, 8), round(re_audit, 8)))
                    print("{}/{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                             format(flag, all, porn_low, porn_high,ph,pl,k, tp,tn,fp,fn,doubt,
                                    round(miss_tp, 12), round(false_alarm, 8), round(acc, 8), round(re_audit, 8)))


if __name__ == '__main__':
    # the threshold should be adjust by facts adn needs to check very carefully before applied online
    porn_the = np.arange(0.2, 0.1, step=0.01)
    porn_the = [round(i, 2) for i in porn_the]

    porn_thh = np.arange(0.9, 0.99, step=0.01)
    porn_thh = [round(i, 2) for i in porn_thh]

    politic_the = np.arange(0.5, 1.11, step=0.01)
    politic_the = [round(i, 2) for i in politic_the]
    politic_the.reverse()

    politic_thh = np.arange(0.1, 0.5, step=0.01)
    politic_thh = [round(i, 2) for i in politic_thh]
    politic_thh.reverse()

    terror_the = np.arange(0.01, 0.1, step=0.01)
    terror_the = [round(i, 2) for i in terror_the]
    print(porn_the.__len__(), porn_the)

    tag = 0
    for po_low in porn_the:
        for po_hi in porn_thh:
            tag += 1
            mp = multiprocessing.Process(target=calculate_metrics2,
                                         args=(po_low, po_hi, politic_the,politic_thh, terror_the, tag))
            mp.start()
