import os
import cv2
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO

print("=" * 60)
print("СЕРВЕР СЧЕТЧИКА ТЕЛЕВИЗОРОВ")
print("=" * 60)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Создаем папки если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/results', exist_ok=True)
print("✓ Папки созданы")

# Загружаем модель
try:
    print("Загрузка модели YOLOv8...")
    model = YOLO('yolov8n.pt')
    print("✓ Модель загружена")
except Exception as e:
    print(f"✗ Ошибка загрузки модели: {e}")
    model = None

# Хранение истории
history = []
previous_count = 0

def detect_tvs(image_path):
    global previous_count
    
    img = cv2.imread(image_path)
    if img is None:
        return None, "Ошибка чтения изображения"
    
    # Оптимизация размера
    h, w = img.shape[:2]
    if max(w, h) > 1280:
        scale = 1280 / max(w, h)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    
    # Детекция
    results = model(img)
    
    # Детектируем телевизоры
    detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes:
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Принимаем телевизоры (72) и стулья (62)
                if cls_id in [62, 72] and conf > 0.25:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': cls_id
                    })
    
    # Рисуем bounding boxes
    result_img = img.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Цвет рамки
        color = (0, 255, 0) if det['class_id'] == 72 else (0, 255, 255)
        
        # Рисуем рамку
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color, 2)
        
        # Текст с уверенностью
        label = f"{conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        
        # Фон для текста
        cv2.rectangle(
            result_img,
            (x1, y1 - text_height - 10),
            (x1 + text_width + 10, y1),
            color,
            -1
        )
        
        # Текст
        cv2.putText(
            result_img,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )
    
    current_count = len(detections)
    change = current_count - previous_count
    
    # Сохраняем результат
    result_path = f"static/results/result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(result_path, result_img)
    
    print(f"📊 БЫЛО: {previous_count}, СТАЛО: {current_count}, ИЗМЕНЕНИЕ: {change:+d}")
    
    # Сохраняем в историю
    history_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'count': current_count,
        'change': change,
        'image_path': result_path
    }
    history.append(history_entry)
    
    # Обновляем для следующего раза
    previous_count = current_count
    
    return {
        'count': current_count,
        'change': change,
        'result_image': result_path
    }, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не загружен'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Файл не выбран'}), 400
    
    # Сохраняем файл
    filename = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    print(f"✓ Файл сохранен: {filename}")
    
    # Обрабатываем изображение
    result, error = detect_tvs(filepath)
    
    if error:
        print(f"✗ Ошибка обработки: {error}")
        return jsonify({'error': error}), 500
    
    print(f"✓ Обнаружено телевизоров: {result['count']}")
    
    return jsonify(result)

@app.route('/history')
def get_history():
    # Возвращаем последние 10 записей
    recent_history = history[-10:] if len(history) > 10 else history.copy()
    return jsonify(recent_history)

@app.route('/stats')
def get_stats():
    if not history:
        return jsonify({'error': 'Нет данных'})
    
    # Берем последние 2 записи
    if len(history) >= 2:
        last = history[-1]
        prev = history[-2]
        return jsonify({
            'was': prev['count'],      # БЫЛО
            'became': last['count'],   # СТАЛО
            'change': last['change']   # ИЗМЕНЕНИЕ
        })
    else:
        # Если только одна запись
        first = history[0]
        return jsonify({
            'was': 0,
            'became': first['count'],
            'change': first['change']
        })

@app.route('/test')
def test():
    return jsonify({
        'status': 'ok',
        'history_length': len(history),
        'previous_count': previous_count
    })

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("СЕРВЕР ЗАПУЩЕН!")
    print("Откройте браузер: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)