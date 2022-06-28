from PIL import Image
from pytesseract import pytesseract

path_to_tesseract = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#Define path to image
path_to_image = r'H:\Video\PyProject\AbnormalFeatures\Images\Defect_1.jpg'

#Point tessaract_cmd to tessaract.exe
pytesseract.tesseract_cmd = path_to_tesseract

#Open image with PIL
img = Image.open(path_to_image)

#Extract text from image
text = pytesseract.image_to_string(img)
Location = text[0 :text.find('m') + 1]

print(text)
