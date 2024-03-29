# tesseract hocr to pandas DataFrame

## pip install tesserhocr2df

### Tested against Windows 10 / Python 3.11 / Anaconda 


```PY
from tesserhocr2df import text2df
from PrettyColorPrinter import add_printer

add_printer(1)
df = text2df(
    img=r"C:\Users\hansc\Desktop\2024-03-29 02_18_09-C__ProgramData_BlueStacks_nxt.png",
    add_after_tesseract_path="",
    add_at_the_end="-l eng+por --psm 3",
    tesseractpath=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    add_imgs=True,
)
print(df)
```