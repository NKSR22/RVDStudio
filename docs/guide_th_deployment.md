# คู่มือการติดตั้งและใช้งานข้ามระบบ (ภาษาไทย)

เอกสารนี้สรุปวิธีใช้งานโปรเจกต์บน `Windows`, `macOS`, และ `Ubuntu` เพื่อเก็บข้อมูล, เทรน, และทดสอบโมเดล ก่อนนำไปใช้กับ Raspberry Pi

## แนวคิดการใช้งาน

เครื่องคอมพิวเตอร์หรือโน้ตบุ๊กใช้สำหรับ:

- เปิด GUI เพื่อดูภาพจากกล้องจริง
- เก็บข้อมูลภาคสนาม
- export dataset
- เทรนและประเมินโมเดล

ส่วน Raspberry Pi ใช้ภายหลังสำหรับ:

- inference จริงบนหุ่นยนต์
- เชื่อมกล้องและ LiDAR
- ตรรกะการตัดสินใจของหุ่นยนต์

## ความต้องการร่วม

- Python `3.10+`
- เว็บแคมที่ระบบมองเห็นได้
- อินเทอร์เน็ตสำหรับติดตั้งแพ็กเกจครั้งแรก
- พื้นที่ดิสก์เพียงพอสำหรับ `.venv`, dataset, และ model

## Windows

### ติดตั้ง

```powershell
cd C:\path\to\OpenCV
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

หรือ:

```powershell
python scripts/setup_venv.py
```

### เปิด GUI

```powershell
.venv\Scripts\Activate.ps1
python -m app.gui
```

### ข้อควรระวัง

- ถ้า PowerShell ไม่ยอม activate ให้ใช้ `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
- หมายเลขกล้องอาจไม่ตรงกับ macOS หรือ Ubuntu
- ถ้ากล้องถูกใช้อยู่โดยโปรแกรมอื่น ให้ปิดโปรแกรมนั้นก่อน

## macOS

### ติดตั้ง

```bash
cd /path/to/OpenCV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

หรือ:

```bash
python scripts/setup_venv.py
```

### เปิด GUI

```bash
source .venv/bin/activate
python -m app.gui
```

### ข้อควรระวัง

- อนุญาตสิทธิ์กล้องให้ Terminal หรือ IDE ใน `System Settings > Privacy & Security > Camera`
- ถ้าเป็น Apple Silicon สามารถใช้เครื่องนี้สำหรับเทรน/ทดสอบเบื้องต้นได้ดี
- หมายเลขกล้องอาจไม่เหมือนระบบอื่น

## Ubuntu

### ติดตั้ง

```bash
cd /path/to/OpenCV
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

หรือ:

```bash
python3 scripts/setup_venv.py
```

### เปิด GUI

```bash
source .venv/bin/activate
python -m app.gui
```

### ข้อควรระวัง

- ถ้า PySide6 หรือ OpenCV เปิดกล้องไม่ได้ อาจต้องติดตั้ง `python3-venv`, `libgl1`, `v4l-utils`
- ถ้า GUI ไม่ขึ้นหรือเจอ error แนว `Could not load the Qt platform plugin "xcb"` ให้รัน `python scripts/diagnose_gui.py` แล้วติดตั้งแพ็กเกจ Ubuntu ที่สคริปต์แนะนำ (ที่เจอบ่อย: `libxcb-cursor0`, `libxcb-xinerama0`, `libxkbcommon-x11-0`, `libglib2.0-0`, `libgl1`)
- ตรวจสอบกล้องด้วย `ls /dev/video*`
- เช็กสิทธิ์การเข้าถึงอุปกรณ์กล้องของ user

## คำสั่งหลักของ workflow

```bash
python scripts/export_yolo_dataset.py --raw data/raw --out datasets/robot_obstacle --clear-out
python scripts/train.py --data configs/dataset.yaml --model models/yolo11n.pt --epochs 50 --imgsz 640
python scripts/evaluate.py --model runs/train/robot_obstacle/weights/best.pt --data configs/dataset.yaml
python scripts/test_model.py --model runs/train/robot_obstacle/weights/best.pt --source 0
```

## แนวทางก่อนลง Raspberry Pi

- เทรนบนเครื่อง desktop หรือ laptop ก่อน
- ทดสอบกับกล้องจริงจากตำแหน่งจริงให้เรียบร้อย
- เมื่อผลแม่นพอแล้วค่อย export/convert ไปใช้บน Pi
- ระหว่างเก็บข้อมูลกับ deploy ต้องรักษาตำแหน่งกล้องให้ใกล้เคียงกัน
