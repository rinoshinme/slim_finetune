import numpy as np


def read_all_scores(score_file):
    porn_scores = []
    politic_scores = []
    violence_scores = []
    ocr_scores = []
    manual_results = []
    with open(score_file, 'r') as f:
        for line in f.readlines():
            if 'null' in line:
                continue
            parts = line.split('\t')

            if len(parts) == 5:
                continue

            manual = parts[5]
            if len(manual.strip()) == 0:
                continue

            try:
                # index = int(parts[0])
                porn = float(parts[1])
                politic = float(parts[2])
                violence = float(parts[3])
                ocr = float(parts[4])
            except:
                # print(parts)
                pass

            porn_scores.append(porn)
            politic_scores.append(politic)
            violence_scores.append(violence)
            ocr_scores.append(ocr)
            manual_results.append(manual.strip())
    return porn_scores, politic_scores, violence_scores, ocr_scores, manual_results


def pvo_label(value, low, high):
    if value < low:
        return 0
    elif value > high:
        return 1
    else:
        return 2


def pol_label(value, low, high):
    if value < low:
        return 1
    elif value > high:
        return 0
    else:
        return 2


def calculate_metrics(data, porn_low, porn_high, politic_low, politic_high, violence_low, violence_high):
    porn_scores, politic_scores, violence_scores, ocr_scores, manual_results = data
    # num_total = len(porn_scores)
    
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    pred_doubt = 0
    normal_labels = ['pass']
    illegal_labels = ['advertise', 'gambling', 'infringement', 'pornography', 'privacy', 'other']
    for porn, politic, violence, ocr, manual in zip(porn_scores, politic_scores, violence_scores,
                                                    ocr_scores, manual_results):
        porn_res = pvo_label(porn, porn_low, porn_high)
        politic_res = pol_label(politic, politic_low, politic_high)
        violence_res = pvo_label(violence, violence_low, violence_high)
        ocr_res = pvo_label(ocr, 0.1, 0.9)

        value = porn_res + politic_res + violence_res + ocr_res
        risk_labels = [porn_res, politic_res, violence_res, ocr_res]
        if manual in normal_labels:
            if value == 0:
                true_neg += 1
            elif value > 0:
                if 1 in risk_labels:
                    false_pos += 1
                elif 2 in risk_labels:
                    pred_doubt += 1
        elif manual in illegal_labels:
            if value == 0:
                false_neg += 1
            elif value > 0:
                if 1 in risk_labels:
                    true_pos += 1
                elif 2 in risk_labels:
                    pred_doubt += 1
    return true_pos, true_neg, false_pos, false_neg, pred_doubt


def write_confs(text_file, idx, porn_low, porn_high, politic_low, politic_high, violence_low, violence_high, 
                tp, tn, fp, fn, doubt, miss, false_alarm, acc, audit):

    values = [idx, porn_low, porn_high, politic_low, politic_high, violence_low, violence_high,
              tp, tn, fp, fn, doubt, miss, false_alarm, acc, audit]
    rep = [str(v) for v in values]
    rep = ','.join(rep)

    with open(text_file, 'a') as f:
        f.write('%s\n' % rep)


def test():
    check_result_file = r'./data/reviewDetail_5weeks.txt'
    summary_file = r'./results/thresholds_summary.csv'

    all_data = read_all_scores(check_result_file)
    total_num = len(all_data[0])

    porn_lows = np.arange(0.001, 0.02, 0.001)
    # porn_highs = np.arange(0.85, 1.00, 0.01)
    porn_highs = np.array([0.85])
    politic_lows = np.arange(0.10, 0.40, 0.02)
    politic_highs = np.arange(0.80, 1.20, 0.02)
    violence_lows = np.arange(0.001, 0.02, 0.001)
    violence_highs = np.array([0.9957])

    total_combs = porn_lows.size * porn_highs.size * politic_lows.size * politic_highs.size * \
        violence_lows.size * violence_highs.size
    print('total combinations = {}'.format(total_combs))
    cnt = 0

    write_confs(summary_file, 'idx', 'porn_low', 'porn_high', 'politic_low', 'politic_high', 
                'violence_low', 'violence_high', 'true_positive', 'true_negative', 
                'false_positive', 'false_negative', 'doubt', 'miss_rate', 'false_alarm', 'accuracy', 'audition')
    # start_time = time.time()
    start_idx = 0
    for porn_tl in porn_lows:
        for porn_th in porn_highs:
            for pol_tl in politic_lows:
                for pol_th in politic_highs:
                    for vio_tl in violence_lows:
                        for vio_th in violence_highs:
                            cnt += 1
                            if cnt < start_idx:
                                continue
                            if cnt % 100 == 0:
                                print('{}/{}'.format(cnt, total_combs))
                                # current_time = time.time()
                                # print(current_time - start_time)
                            tp, tn, fp, fn, doubt = calculate_metrics(all_data, porn_tl, porn_th,
                                                                      pol_tl, pol_th, vio_tl, vio_th)
                            miss = fn / total_num
                            fa = fp / total_num
                            acc = (tp + tn) / total_num
                            audit = doubt / total_num
                            # miss + fa + acc + audit = 1.

                            write_confs(summary_file, cnt, porn_tl, porn_th, pol_tl, pol_th, vio_tl, vio_th, 
                                        tp, tn, fp, fn, doubt, miss, fa, acc, audit)


if __name__ == '__main__':
    test()
