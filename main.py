import cv2
from face_detector import FaceDetector

def run_webcam_face_detection():
    """
    Web kamerasını açar, FaceDetector sınıfını kullanarak yüzleri algılar
    ve sonucu ekranda gösterir.
    """
    
    try:
        detector = FaceDetector()
    except IOError as e:
        print(e)
        print("Program sonlandırılıyor.")
        return

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Hata: Web kamerası başlatılamadı.")
        return
        
    print("Web kamerası açıldı. Çıkmak için 'q' tuşuna basın...")

    while True:

        ret, frame = cap.read()
        
        if not ret:
            print("Hata: Kameradan kare okunamadı. Akış sonlanıyor.")
            break

        
        faces, processed_frame = detector.detect(frame)

        cv2.putText(processed_frame, f'Algilanan Yuz: {len(faces)}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    
        cv2.imshow('OOP Yuz Algilama - Cikis icin "q" basin', processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    print("Kamera kapatılıyor ve pencereler temizleniyor...")
    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam_face_detection()