import matplotlib.pyplot as plt


def read_logs(log_file):
    val_losses = []
    val_accs = []

    with open(log_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('Evaluation'):
                next_line = f.readline()
                next_line = next_line.strip()
                fields = next_line.split(' ')
                print(fields)
                val_losses.append(float(fields[4][:-1]))
                val_accs.append(float(fields[6]))

    return val_losses, val_accs


if __name__ == '__main__':
    logf = r'D:\temp\logs.txt'
    losses, accs = lines = read_logs(logf)
    steps = [i * 100 for i in range(len(losses))]
    plt.plot(steps, losses)
    plt.plot(steps, accs)
    plt.grid()
    plt.show()
