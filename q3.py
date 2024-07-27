import cv2

img=cv2.imread('/home/student/PycharmProjects/220962408_CV/images/flower.jpg')

print(img.shape)
height,width,channels=img.shape
print(f"Height: {height}")
print(f"Width: {width}")
print(f"Channels: {channels}")

x=int(input("Enter x value: "))
y=int(input("Enter y value: "))

print(f"[b,g,r]={img[x,y]}")
