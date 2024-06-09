import json
import matplotlib.pyplot as plt

def plot_and_save(data, filename='plot.png'):
    plt.figure()
    plt.plot(data)
    plt.title('Train Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.savefig(filename)
    plt.cla()


def load_jsonl(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data


def convert_answer(answer):
    converted = []
    missed = []
    for a in answer:
        if a.lower() == 'yes':
            converted.append(1)
        elif a.lower() == 'no':
            converted.append(0)
        else:
            missed.append(a)
    return converted, missed