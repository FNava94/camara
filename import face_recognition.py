import face_recognition
import picamera
import numpy as np

def capture_image(file_path='captured_image.jpg'):
    with picamera.PiCamera() as camera:
        camera.resolution = (640, 480)
        camera.capture(file_path)

def recognize_face(known_faces, unknown_face):
    for name, known_face_encoding in known_faces.items():
        matches = face_recognition.compare_faces([known_face_encoding], unknown_face)
        if True in matches:
            return name
    return "Desconocido"

def main():
    # Captura una imagen con la cámara
    capture_image()

    # Carga las imágenes conocidas y sus codificaciones faciales
    known_faces = {
        "Persona1": face_recognition.face_encodings(face_recognition.load_image_file("persona1.jpg"))[0],
        # Agrega más personas según sea necesario
    }

    # Carga la imagen recién capturada
    unknown_image = face_recognition.load_image_file("captured_image.jpg")

    # Encodifica la cara de la imagen desconocida
    unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

    # Realiza el reconocimiento facial
    recognized_person = recognize_face(known_faces, unknown_face_encoding)

    # Imprime el resultado
    print(f"La persona reconocida es: {recognized_person}")

if __name__ == "__main__":
    main()
