import cv2 as cv
import os

# ================= اختيار الوضع =================
mode = 'webcam'   # image / video / webcam
path = 'photo.jpg'  # للصورة أو الفيديو

# ================= face detector =================
face_cascade = cv.CascadeClassifier(
    cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def process_img(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4)

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        blur = cv.GaussianBlur(face, (99, 99), 30)
        img[y:y+h, x:x+w] = blur

    return img


# ================= output folder =================
out_dir = './output'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


# ================= IMAGE =================
if mode == 'image':

    img = cv.imread(path)

    if img is None:
        print("الصورة مش موجودة ❌")
    else:
        output = process_img(img)

        cv.imshow('Output', output)
        cv.imwrite(os.path.join(out_dir, 'output.jpg'), output)

        cv.waitKey(0)
        cv.destroyAllWindows()
        print("Image saved ✅")


# ================= VIDEO =================
elif mode == 'video':

    cap = cv.VideoCapture(path)

    ret, frame = cap.read()

    out = cv.VideoWriter(
        os.path.join(out_dir, 'output.mp4'),
        cv.VideoWriter_fourcc(*'mp4v'),
        25,
        (frame.shape[1], frame.shape[0])
    )

    while ret:
        output = process_img(frame)
        out.write(output)

        cv.imshow('Video', output)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        ret, frame = cap.read()

    cap.release()
    out.release()
    cv.destroyAllWindows()
    print("Video saved ✅")


# ================= WEBCAM =================
elif mode == 'webcam':

    cap = cv.VideoCapture(0)

    recording = False

    out = cv.VideoWriter(
        os.path.join(out_dir, 'webcam.mp4'),
        cv.VideoWriter_fourcc(*'mp4v'),
        25,
        (640, 480)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔥 هنا الحل: إلغاء المراية
        frame = cv.flip(frame, 1)

        output = process_img(frame)

        # تسجيل الفيديو لو شغال
        if recording:
            out.write(output)

        cv.imshow('Webcam', output)

        key = cv.waitKey(1) & 0xFF

        # 📸 حفظ صورة
        if key == ord('s'):
            cv.imwrite(os.path.join(out_dir, 'snapshot.jpg'), output)
            print("Snapshot saved 📸")

        # 🎥 تشغيل/إيقاف تسجيل
        elif key == ord('r'):
            recording = not recording
            print("Recording..." if recording else "Stopped recording")

        # ❌ خروج
        elif key == ord('q'):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()