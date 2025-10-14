from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe.val_pe_free import YOLOEPEFreeDetectValidator
import json
import os

unfused_model = YOLOE("yoloe-v8l.yaml").cuda()
unfused_model.load("yoloe-v8l-seg.pt")
unfused_model.eval()

with open('ram_tag_list.txt', 'r') as f:
    names = [x.strip() for x in f.readlines()]
vocab = unfused_model.get_vocab(names)

model = YOLOE("yoloe-v8l-seg-pf.pt").cuda()
model.set_vocab(vocab, names=names)
model.model.model[-1].is_fused = True
model.model.model[-1].conf = 0.001
model.model.model[-1].max_det = 1000

results = model.predict('input_images/xylophone.jpg', save=True, save_dir='output')

# Extraer etiquetas únicas de los objetos detectados
unique_labels = set()
for result in results:
    if result.boxes is not None and len(result.boxes) > 0:
        # Obtener los índices de las clases detectadas
        class_ids = result.boxes.cls.cpu().numpy()
        # Convertir índices a nombres de clases
        for class_id in class_ids:
            label = names[int(class_id)]
            unique_labels.add(label)

# Crear directorio output si no existe
os.makedirs('output', exist_ok=True)

# Guardar las etiquetas únicas en un archivo JSON
output_data = {
    "semantic_labels": sorted(list(unique_labels))
}

with open('output/semantic_labels.json', 'w') as f:
    json.dump(output_data, f, indent=2)

print(f"Detected {len(unique_labels)} unique semantic labels: {sorted(list(unique_labels))}")
