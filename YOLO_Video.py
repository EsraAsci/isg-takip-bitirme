from ultralytics import YOLO
import cv2
import math
import torch
from playsound import playsound
import threading
import os
import time


def play_sound(sound_file, played_sounds):
    if sound_file not in played_sounds:
        played_sounds.add(sound_file)
        threading.Thread(target=playsound, args=(sound_file,), daemon=True).start()

def log_error(error_message):
    """Belirli bir hata türünü ve ilgili IP adresini bir dosyaya kaydeder."""
    desktop_path = os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')
    log_file_path = os.path.join(desktop_path, "Kayıt.txt")
    with open(log_file_path, "a") as file:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"{timestamp}: {error_message} \n")
def video_detection(path_x):
    video_capture = path_x
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('cihaz = ', device)

    cap = cv2.VideoCapture(video_capture)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    model = YOLO("YOLO-Weights/ppe.pt").to(device)
    sinif_isimleri = ['baretsiz', 'beyaz-baret', 'iş-ayakkabısı', 'kirmizi-baret', 'kirmizi-yelek',
                      'mavi-baret', 'mavi-yelek', 'normal-ayakkabı', 'sari-baret', 'sari-yelek',
                      'turuncu-baret', 'turuncu-yelek', 'yeleksiz']

    played_sounds = set()  # Set to track played sounds

    while True:
        success, img = cap.read()
        if not success:
            break

        sonuclar = model(img, stream=True)
        baretsiz_var = False
        yeleksiz_var = False
        baret_var = False
        yelek_var = False

        for r in sonuclar:
            kutular = r.boxes
            for kutu in kutular:
                x1, y1, x2, y2 = kutu.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((kutu.conf[0] * 100)) / 100
                index = int(kutu.cls[0])
                sinif_adi = sinif_isimleri[index]
                label = f'{sinif_adi}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                renk = (0, 204, 255) if sinif_adi == 'baretsiz' else \
                    (222, 82, 175) if sinif_adi == 'beyaz-baret' else \
                        (128, 0, 128) if sinif_adi == 'iş-ayakkabısı' else \
                            (124, 252, 0) if sinif_adi == 'kirmizi-baret' else \
                                (255, 102, 102) if sinif_adi == 'kirmizi-yelek' else \
                                    (255, 255, 102) if sinif_adi == 'mavi-baret' else \
                                        (255, 165, 0) if sinif_adi == 'mavi-yelek' else \
                                            (0, 0, 128) if sinif_adi == 'normal-ayakkabı' else \
                                                (255, 155, 70) if sinif_adi == 'sari-baret' else \
                                                    (255, 182, 193) if sinif_adi == 'sari-yelek' else \
                                                        (128, 128, 128) if sinif_adi == 'turuncu-baret' else \
                                                            (222, 122, 175) if sinif_adi == 'turuncu-yelek' else \
                                                                (0, 149, 255) if sinif_adi == 'yeleksiz' else (
                                                                85, 45, 255)

                if conf > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), renk, 3)
                    cv2.rectangle(img, (x1, y1), c2, renk, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

                if sinif_adi == 'baretsiz':
                    baretsiz_var = True
                if sinif_adi == 'yeleksiz':
                    yeleksiz_var = True
                if 'baret' in sinif_adi and sinif_adi != 'baretsiz':
                    baret_var = True
                if 'yelek' in sinif_adi and sinif_adi != 'yeleksiz':
                    yelek_var = True

        y_offset = frame_height - 50  # Y offset for text at the bottom
        text_x = 50  # X coordinate for text

        if baretsiz_var and yeleksiz_var:
            cv2.putText(img, "Lutfen ekipmanlarinizi takiniz", (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            play_sound('sounds/ekipman.mp3', played_sounds)
            error_message='ekipman tespit edilemedi.'
            log_error(error_message)
        elif baretsiz_var:
            cv2.putText(img, "Lutfen baretinizi takiniz", (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            play_sound('sounds/baret (2).mp3', played_sounds)
            error_message = 'baret tespit edilemedi.'
            log_error(error_message)
        elif yeleksiz_var:
            cv2.putText(img, "Lutfen yeleginizi giyiniz", (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)
            play_sound('sounds/yelek (2).mp3', played_sounds)
            error_message = 'yelek tespit edilemedi.'
            log_error(error_message)
        elif baret_var and yelek_var:
            cv2.putText(img, "Ekipmanlariniz tamdir, gecebilirsiniz", (text_x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2, cv2.LINE_AA)
            play_sound('sounds/tam (2).mp3', played_sounds)
            error_message = 'Başarılı giriş yapıldı'
            log_error(error_message)

        yield img


cv2.destroyAllWindows()
