import numpy as np
import copy

def RandomSwap(in_report,alpha=0.1):
    """
    Randomly swap the word, in_report is not changed
    """
    report = copy.deepcopy(in_report)
    time = int(len(report)*alpha/2)
    if time < 1:
        return report
    for i in range(time):
        word_1_index = int(np.random.random()*len(report))
        word_2_index = int(np.random.random()*len(report))
        x = report[word_1_index]
        report[word_1_index] = report[word_2_index]
        report[word_2_index] = x
    return report

def RandomDelete(report,alpha=0.1):
    """
    Randomly delete the word
    """
    new_report = []
    for word in report:
        if np.random.random() < alpha:
            continue
        new_report.append(word)
    return new_report
