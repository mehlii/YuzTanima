import cv2

class FaceDetector:
    """
    Bu sınıf, OpenCV'nin önceden eğitilmiş Haar Cascade modelini kullanarak
    bir görüntü karesindeki (frame) yüzleri algılamak için kullanılır.
    """

    def __init__(self):
        """
        Sınıftan bir nesne oluşturulduğunda çalışan kurucu metottur.
        Yüz algılama modelini burada yükleriz.
        """

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        

        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        

        if self.face_cascade.empty():
            raise IOError(f"Hata: Haar Cascade modeli yüklenemedi: {cascade_path}")
        print("FaceDetector sınıfı başlatıldı ve model yüklendi.")


    def detect(self, frame, draw_rectangles=True):
        """
        Verilen bir video karesinde (frame) yüzleri algılar.
        
        :param frame: İşlenecek video karesi (BGR formatında)
        :param draw_rectangles: Yüzlerin etrafına dikdörtgen çizilip çizilmeyeceği
        :return: Yüzlerin koordinatları (x, y, w, h) listesi VE (istenirse) işlenmiş kare
        """

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        

        faces = self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,     
            minNeighbors=5,      
            minSize=(30, 30)     
        )
        

        if draw_rectangles:

            for (x, y, w, h) in faces:

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return faces, frame