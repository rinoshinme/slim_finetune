import os
import matplotlib.pyplot as plt


class_names = ['normal', 'riot', 'crash', 'fire', 'army', 'terrorism', 'weapon', 'bloody', 'protest']
ng_indices = [1, 2, 3, 5, 7, 8, ]
ok_indices = [0, 4, 6]


def read_scores(text_file):
    labels = []
    scores = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            parts = line.split('\t')
            score_values = [float(v) for v in parts[1].split(',')]
            label = os.path.split(parts[0])[0].split('\\')[-1]
            labels.append(label)
            scores.append(score_values)
    return labels, scores


def get_nums(labels, scores, threshold):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    # normal is positive
    # ng is negative
    for label, score in zip(labels, scores):
        lbl_idx = class_names.index(label)
        ng_score = sum(score[v] for v in ng_indices)
        if lbl_idx in ok_indices and ng_score <= threshold:
            tp += 1
        elif lbl_idx in ok_indices and ng_score > threshold:
            fn += 1
        elif lbl_idx in ng_indices and ng_score > threshold:
            tn += 1
        else:
            fp += 1
    return tp, fp, tn, fn


def calc_metrics(tp, fp, tn, fn):
    miss_rate = fp * 1.0 / (tn + fp)
    false_alarm = fn * 1.0 / (tp + fn)
    accuracy = 1.0 - (fp + fn) * 1.0 / (tp + fp + tn + fn)
    # accuracy = 1.0 - accuracy
    return miss_rate, false_alarm, accuracy


def test():
    txt = r'D:\data\21cn_test_data\result.txt'
    labels, scores = read_scores(txt)
    tp, fp, tn, fn = get_nums(labels, scores, threshold=0.6)
    miss_rate, false_alarm, accuracy = calc_metrics(tp, fp, tn, fn)

    print('miss rate = {}%'.format(miss_rate * 100))
    print('false alarm = {}%'.format(false_alarm * 100))
    print('accuracy = {}%'.format(accuracy * 100))


def test_curve():
    txt = r'D:\data\21cn_test_data\result.txt'
    labels, scores = read_scores(txt)
    thresholds = [i * 0.01 for i in range(40, 101)]
    mrs = []
    fas = []
    accs = []
    for th in thresholds:
        tp, fp, tn, fn = get_nums(labels, scores, threshold=th)
        miss_rate, false_alarm, accuracy = calc_metrics(tp, fp, tn, fn)
        mrs.append(miss_rate)
        fas.append(false_alarm)
        accs.append(accuracy)

    for th, mr, fa, acc in zip(thresholds, mrs, fas, accs):
        print('{}: {} - {} - {}'.format(th, mr, fa, acc))

    plt.plot(thresholds, mrs, 'k*-', label='miss_rate')
    plt.plot(thresholds, fas, 'b-', label='false alarm')
    plt.plot(thresholds, accs, 'r-', label='accuracy')
    plt.grid()
    plt.legend()
    plt.xlabel('thresholds')
    plt.ylabel('metrics')
    plt.show()


if __name__ == '__main__':
    # test()
    test_curve()
