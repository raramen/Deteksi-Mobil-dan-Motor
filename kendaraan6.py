import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5.uic import loadUi

# Inisialisasi posisi garis pemisah dan offset
pos_line = 300
offset = 1.5

# Membuat kelas QMainWindow untuk menampilkan gambar
class showimage(QMainWindow):
    def __init__(self):
        super(showimage, self).__init__()
        loadUi('Gui2.ui', self)  # Memuat file GUI dari Gui2.ui
        self.video = cv2.VideoCapture('video.mp4')  # Membuka video
        _, bg = self.video.read()  # Membaca frame pertama untuk latar belakang
        self.count_motor = 0  # Variabel untuk menghitung jumlah motor
        self.count_mobil = 0  # Variabel untuk menghitung jumlah mobil

    def start(self):
        # Mendefinisikan koordinat dan ukuran area yang diminati
        x1, y1, w1, h1 = 0, 0, 300, 30

        # Looping untuk memproses setiap frame video
        while True:
            ret, frame = self.video.read()  # Membaca frame dari video

            if ret:  # Jika pembacaan frame berhasil
                # Konversi ke citra grayscale
                gray_frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

                # Menampilkan citra grayscale di label_11
                shape_gray = self.show_image_in_label(gray_frame, self.label_11)

                # Filter Gaussian untuk mengurangi noise
                kernel_size = 25
                sigma = 1.5
                kernel = cv2.getGaussianKernel(kernel_size, sigma)
                kernel = np.outer(kernel, kernel.transpose())
                kernel /= kernel.sum()
                filtered_frame = cv2.filter2D(gray_frame, -1, kernel)

                # Menampilkan hasil filter Gaussian di label_12
                shape_filtered = self.show_image_in_label(filtered_frame, self.label_12)

                # Melakukan thresholding untuk segmentasi
                ret, thresh = cv2.threshold(filtered_frame, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                kernel1 = np.ones((30, 30), np.uint8)
                closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel1)

                # Menampilkan hasil thresholding di label_13
                shape_thresh = self.show_image_in_label(thresh, self.label_13)
                
                # Menampilkan hasil closing di label_15
                shape_closing = self.show_image_in_label(closing, self.label_15)

                # Melakukan dilasi untuk menghubungkan area yang berdekatan
                kernel2 = np.ones((5, 5), np.uint8)
                dilation = cv2.dilate(closing, kernel2, iterations=5)

                # Menampilkan hasil dilasi di label_16
                shape_dilation = self.show_image_in_label(dilation, self.label_16)

                # Mendeteksi kontur
                contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.line(frame, (0, pos_line), (1000, pos_line), (255, 255, 255), 1)

                motor_count_per_frame = 0
                mobil_count_per_frame = 0

                # Iterasi melalui setiap kontur yang dideteksi
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    luas = cv2.contourArea(contour)

                    # Jika kontur berada di atas atau di bawah garis pemisah
                    if y <= (pos_line + offset) and luas >= 10000 and luas <= 20000 and y >= (pos_line - offset):
                        # Mendeteksi sebagai motor
                        cv2.putText(frame, 'Motor', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        helm = frame[y:y + h, x:x + w]
                        filename = f'objek.png'
                        cv2.imwrite(filename, helm)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        RGB_img = cv2.imread(filename)
                        motor_count_per_frame += 1  # Menambahkan hitungan untuk motor dalam frame ini
                        self.baca_gambar(RGB_img)
                        self.count_motor += 0.5  # Menambahkan hitungan total untuk motor

                    # Jika kontur berada di atas atau di bawah garis pemisah dan memiliki luas yang berbeda
                    elif y < (pos_line + offset) and luas > 20000 and luas < 80000 and y > (pos_line - offset):
                        # Mendeteksi sebagai mobil
                        cv2.putText(frame, 'Mobil', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        helm = frame[y:y + h, x:x + w]
                        filename = f'objek.png'
                        cv2.imwrite(filename, helm)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
                        RGB_img = cv2.imread(filename)
                        crop_img = RGB_img[y1:y1 + h1, x1:x1 + w1]
                        mobil_count_per_frame += 1  # Menambahkan hitungan untuk mobil dalam frame ini
                        self.baca_gambar(RGB_img)
                        self.count_mobil += 0.5  # Menambahkan hitungan total untuk mobil

                # Konversi frame OpenCV ke format QImage
                qformat = QImage.Format_Indexed8
                if len(frame.shape) == 3:
                    if (frame.shape[2]) == 4:
                        qformat = QImage.Format_RGBA8888
                    else:
                        qformat = QImage.Format_RGB888
                video = QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0], qformat)
                video = video.rgbSwapped()

                # Menampilkan video di label
                self.label.setPixmap(QPixmap.fromImage(video))
                self.label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                self.label.setScaledContents(True)

                # Menampilkan hitungan terbaru di label
                self.label_9.setText(str(int(self.count_motor)))  # Menampilkan hitungan motor di label_9
                self.label_10.setText(str(int(self.count_mobil)))  # Menampilkan hitungan mobil di label_10

            # Hentikan perulangan saat video selesai
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Metode untuk menampilkan citra di label_2
    def baca_gambar(self, RGB_img):
        qformat1 = QImage.Format_Indexed8

        if len(RGB_img.shape) == 3:
            if (RGB_img.shape[2]) == 4:
                qformat1 = QImage.Format_RGBA8888
            else:
                qformat1 = QImage.Format_RGB888

        # Konversi citra OpenCV ke QImage
        Cvt2qt = QImage(RGB_img, RGB_img.shape[1], RGB_img.shape[0], RGB_img.strides[0], qformat1)
        Cvt2qt = Cvt2qt.rgbSwapped()

        # Menampilkan citra di label_2
        self.label_2.setPixmap(QPixmap.fromImage(Cvt2qt))
        self.label_2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        self.label_2.setScaledContents(True)

    # Metode untuk menampilkan citra di label
    def show_image_in_label(self, image, label):
        qformat = QImage.Format_Indexed8
        if len(image.shape) == 3:
            if (image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img = QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        img = img.rgbSwapped()

        # Menampilkan citra di label
        label.setPixmap(QPixmap.fromImage(img))
        label.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        label.setScaledContents(True)

        # Mengembalikan bentuk gambar sebagai tuple (height, width, channels)
        return image.shape

# Membuat instance dari aplikasi Qt
app = QtWidgets.QApplication(sys.argv)
window = showimage()
window.setWindowTitle('Project Akhir')
window.show()
window.start()  # Memulai pemrosesan video
sys.exit(app.exec_())  # Keluar dari aplikasi saat jendela ditutup