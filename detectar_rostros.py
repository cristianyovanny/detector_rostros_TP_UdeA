import cv2

#crear la cascada de haar
cc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Cargar la imagen
img_original = cv2.imread("personas-sociables.jpg") # imagen tomada de https://curiosidadintelligent.blogspot.com/2017/03/ciudad-mas-sociable-del-mundo.html"
img_grises = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

#detectar los rostros en la img_original
rostros = cc.detectMultiScale(
    img_grises,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Se encontraron {0} rostros!".format(len(rostros)))


#Dibuja un rectangulo alrededor de los rostros

for (x,y,w,h) in rostros:
    cv2.rectangle(img_original, (x, y), (x + w, y + h), (0, 255, 0),2)

cv2.imshow("rostros encontrados", img_original)
cv2.waitKey(0)

