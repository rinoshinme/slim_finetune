import cv2
import os


def extract_video(video_filename, save_folder, save_step=25):
    capture = cv2.VideoCapture(video_filename)
    base_name = os.path.split(video_filename)[-1].split('.')[0]
    cnt = 0
    while True:
        # print(cnt)
        ret, frame = capture.read()
        cnt += 1
        if not ret:
            break
        if cnt % save_step != 0:
            continue

        save_path = os.path.join(save_folder, '%s_%06d.jpg' % (base_name, cnt))
        print(save_path)
        if os.path.exists(save_path):
            continue
        cv2.imwrite(save_path, frame)


def read_logfile(logfile):
    names = []
    if not os.path.exists(logfile):
        f = open(logfile, 'w')
        f.close()
    with open(logfile, 'r') as logf:
        for line in logf.readlines():
            names.append(line.strip())
    return names


def write_logfile(logfile, name):
    with open(logfile, 'a+') as logf:
        logf.write('%s\n' % name)


def extract_batch(video_folder, target_folder, log_file, save_step=10):
    names = read_logfile(log_file)
    for root, dirs, files in os.walk(video_folder):
        for f in files:
            if not f.lower().endswith('.mp4'):
                continue
            video_file = os.path.join(root, f)
            if video_file in names:
                continue
            write_logfile(log_file, video_file)
            print('processing {}'.format(video_file))
            extract_video(video_file, target_folder, save_step=save_step)


if __name__ == '__main__':
    target_folder = r'E:\dataset\bloody_frames'
    vid_folder = r'D:\data\bloody_video'
    logfile = r'D:\data\bloody_video\extract_log.txt'
    extract_batch(vid_folder, target_folder, logfile)
