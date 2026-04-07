# คู่มือ Export / เทรน / ทดสอบ (ภาษาไทย)

เอกสารนี้อธิบายขั้นตอนหลังจากเราเก็บข้อมูลด้วย GUI แล้ว: แปลงข้อมูลเป็นชุดข้อมูล YOLO, เทรน, ประเมินผล, และทดสอบกับกล้องจริง

ถ้าคุณใช้งานบนหลายระบบปฏิบัติการ ให้ดูคู่มือเพิ่มที่ `docs/guide_th_deployment.md`

## 1) เปิด GUI เพื่อเก็บข้อมูล

รัน:

`macOS / Ubuntu`

```bash
cd /path/to/RVDStudio
source .venv/bin/activate
python -m app.gui
```

`Windows PowerShell`

```bash
cd /path/to/RVDStudio
.venv\Scripts\Activate.ps1
python -m app.gui
```

กด `Capture Frame` เพื่อบันทึกข้อมูลลง `data/raw/`

หลังจากเก็บข้อมูลแล้ว ให้ไปที่แท็บ `Review / Label` ใน GUI เพื่อ:

- เปิดภาพที่เก็บไว้ตาม session
- ดูผล detect เดิม
- ลากเมาส์บนภาพเพื่อวาด `bbox` ใหม่ได้โดยตรง
- เพิ่ม / ลบ / แก้ไข `label`, `confidence`, `bbox`
- บันทึกกลับเป็น `corrected_detections` ในไฟล์ metadata

## 2) Export เป็นชุดข้อมูล YOLO

คุณสามารถใช้แท็บ `Train / Evaluate` ใน GUI เพื่อ export dataset ได้โดยตรง หรือจะใช้สคริปต์ `scripts/export_yolo_dataset.py` เพื่อแปลง `data/raw/` ไปเป็น YOLO dataset ใน `datasets/robot_obstacle/` ก็ได้

รัน (แนะนำเริ่มต้น):

`macOS / Ubuntu`

```bash
source .venv/bin/activate
python scripts/export_yolo_dataset.py --raw data/raw --out datasets/robot_obstacle --clear-out
```

`Windows PowerShell`

```bash
.venv\Scripts\Activate.ps1
python scripts/export_yolo_dataset.py --raw data/raw --out datasets/robot_obstacle --clear-out
```

ผลลัพธ์จะได้โครงสร้างประมาณนี้:

```text
datasets/robot_obstacle/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

### class mapping ของโปรเจกต์นี้

- `0 = person`
- `1 = obstacle`

โหมด label:

- `--label-mode prelabel` (ค่าเริ่มต้น) จะใช้ detection ในไฟล์ meta JSON มาสร้าง label อัตโนมัติ
- ถ้ามี `corrected_detections` จากหน้า `Review / Label` ระบบ export จะใช้ค่าที่แก้แล้วก่อน
- `--label-mode empty` จะสร้างไฟล์ label ว่าง เพื่อให้คุณไป label แบบ manual ด้วยเครื่องมืออื่น

ถ้าต้องการคัดกรอง detection ที่ความมั่นใจต่ำ:

```bash
python scripts/export_yolo_dataset.py --min-conf 0.35
```

## 3) เทรนโมเดล

คุณสามารถเริ่มเทรนจากแท็บ `Train / Evaluate` ใน GUI หรือใช้สคริปต์ที่เตรียมไว้:

`macOS / Ubuntu`

```bash
source .venv/bin/activate
python scripts/train.py --data configs/dataset.yaml --model models/yolo11n.pt --epochs 50 --imgsz 640
```

`Windows PowerShell`

```bash
.venv\Scripts\Activate.ps1
python scripts/train.py --data configs/dataset.yaml --model models/yolo11n.pt --epochs 50 --imgsz 640
```

น้ำหนักโมเดลที่ดีที่สุดโดยปกติจะอยู่ใน:

```text
runs/train/robot_obstacle/weights/best.pt
```

หมายเหตุ: ถ้าเครื่องคุณมี GPU ที่รองรับ macOS (MPS) หรือ CUDA (ถ้ามี) Ultralytics จะพยายามเลือกให้เอง

## 4) ประเมินผล (Validation)

`macOS / Ubuntu`

```bash
source .venv/bin/activate
python scripts/evaluate.py --model runs/train/robot_obstacle/weights/best.pt --data configs/dataset.yaml
```

`Windows PowerShell`

```bash
.venv\Scripts\Activate.ps1
python scripts/evaluate.py --model runs/train/robot_obstacle/weights/best.pt --data configs/dataset.yaml
```

สรุปผลจะถูกบันทึกไว้ที่:

```text
runs/eval/robot_obstacle_eval/summary.json
```

## 5) ทดสอบกับกล้องจริง หรือวิดีโอ

ทดสอบแบบ live webcam:

`macOS / Ubuntu`

```bash
source .venv/bin/activate
python scripts/test_model.py --model runs/train/robot_obstacle/weights/best.pt --source 0
```

`Windows PowerShell`

```bash
.venv\Scripts\Activate.ps1
python scripts/test_model.py --model runs/train/robot_obstacle/weights/best.pt --source 0
```

ทดสอบกับไฟล์วิดีโอ:

```bash
python scripts/test_model.py --model runs/train/robot_obstacle/weights/best.pt --source path/to/video.mp4
```

## ข้อควรระวังสำคัญ

- การ export แบบ prelabel จะช่วยเริ่มเร็ว แต่ label อาจผิด ต้องมีขั้นตอนตรวจและแก้ในรอบต่อไป
- ถ้าคุณต้องการคุณภาพโมเดลดีขึ้นจริง ให้เพิ่ม “manual labeling” หรือ “review prelabels” ในชุดข้อมูลที่สำคัญ
- หลีกเลี่ยงการเทรนด้วยข้อมูลที่ซ้ำมากเกินไป (เฟรมคล้ายกันมาก) เพราะจะเสียเวลาและไม่ช่วยความแม่นเท่าที่ควร
