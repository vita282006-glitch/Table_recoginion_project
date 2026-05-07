import cv2
import numpy as np
from ultralytics import YOLO
import json

class TableCellDetector:
    def __init__(self, model_path='best.pt'):
        self.model = YOLO(model_path)
    
    def detect_cells(self, image_path, conf_threshold=0.25):
        """
        Детектирует ячейки и возвращает координаты
        """
        # Загрузка и предобработка изображения
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        height, width = image.shape[:2]
        
        # Инференс
        results = self.model(image, conf=conf_threshold)
        
        # Извлечение координат
        cells = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cells.append({
                        'x1': int(x1),
                        'y1': int(y1), 
                        'x2': int(x2),
                        'y2': int(y2),
                        'width': int(x2 - x1),
                        'height': int(y2 - y1),
                        'confidence': float(box.conf[0])
                    })
        
        return {
            'image_path': image_path,
            'image_size': {'width': width, 'height': height},
            'total_cells': len(cells),
            'cells': cells
        }
    
    def visualize(self, image_path, output_path='output.jpg'):
        """
        Визуализация результатов
        """
        image = cv2.imread(image_path)
        result = self.detect_cells(image_path)
        
        for cell in result['cells']:
            cv2.rectangle(image, 
                         (cell['x1'], cell['y1']),
                         (cell['x2'], cell['y2']),
                         (0, 255, 0), 2)
            # Добавляем подпись с уверенностью
            cv2.putText(image, f"{cell['confidence']:.2f}",
                       (cell['x1'], cell['y1']-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, image)
        print(f"Результат сохранён в {output_path}")
        
        return result

# Использование
detector = TableCellDetector('model_table_cells.pt')
result = detector.detect_cells('document_scan.jpg')

# Вывод результата
print(json.dumps(result, indent=2))

# Визуализация
detector.visualize('document_scan.jpg', 'detected_cells.jpg')

# Сохранение в JSON
with open('cells_coordinates.json', 'w') as f:
    json.dump(result, f, indent=2)