# get video ,open video, get photo
import cv2
import numpy as np

capID=0
out_videofile='outlogi.avi'
open_videofile='outpy.avi'
picfile="test1.jpg"

print('c:录制视频\no:打开视频播放\np:拍照')
n = input("Please input c、o or p:")
if n=='c':
  cap = cv2.VideoCapture(capID)
  # Check if camera opened successfully
  if (cap.isOpened() == False):
    print("Unable to read camera feed")
  # 默认分辨率取决于系统。
  # 我们将分辨率从float转换为整数。
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
  # 定义编解码器并创建VideoWriter对象。输出存储在“outpy.avi”文件中。
  out = cv2.VideoWriter(out_videofile, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))
  while(True):
    ret, frame = cap.read()

    if ret == True:

      # Write the frame into the file 'output.avi'
      out.write(frame)

      # Display the resulting frame
      cv2.imshow('frame',frame)

      # Press Q on keyboard to stop recording
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      break
  cap.release()
  out.release()

elif n=='o':
  cap1 = cv2.VideoCapture(open_videofile)

  if (cap1.isOpened() == False):
    print("Error opening video stream or file")
  while (cap1.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap1.read()
    if ret == True:

      # Display the resulting frame
      cv2.imshow('Frame', frame)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Break the loop
    else:
      print('play video ending')
      break
  # When everything done, release the video capture object
  cap1.release()
# 拍照
elif n=='p':
  print('在显示界面按下o表示保存图片')
  cap = cv2.VideoCapture(capID)
  # Check if camera opened successfully
  if (cap.isOpened() == False):
    print("Unable to read camera feed")
  while (1):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("pic", gray)
    cv2.imshow("pic", frame)

    if cv2.waitKey(1) & 0xFF == ord('o'):
        cv2.imwrite(picfile, gray)
        print("已存储：%s" % picfile)
        break
# When everything done, release the video capture and video write objects

# Closes all the frames
cv2.destroyAllWindows()
