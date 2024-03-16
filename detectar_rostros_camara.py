import cv2

#crear la cascada de haar
cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

vc = cv2.VideoCapture(0)

while True:
    #Capturar diapositiva por diapositiva
    ret, frame = vc.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostros = cc.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #Dibujar un rectangulo alrededor de los rotros
    for (x,y,w,h) in rostros:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),2)

    #Muestra el resultado
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#Liberar la memoria de la captura de video
vc.release()
cv2.destroyAllWindows()
