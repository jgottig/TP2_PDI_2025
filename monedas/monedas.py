import cv2
import numpy as np
import matplotlib.pyplot as plt

#Definimos funciones necesarias para mostrar los pasos de forma ordenada
def show_image(window_name, image):

    # Parámetros de visualización (manteniendo los originales)
    max_display_width = 1000
    max_display_height = 800

    h, w = image.shape[:2]
    
    if w > max_display_width or h > max_display_height:
        scale = min(max_display_width / w, max_display_height / h)
        display_w = int(w * scale)
        display_h = int(h * scale)
        display_img = cv2.resize(image, (display_w, display_h), interpolation=cv2.INTER_AREA) #Reduce
    else:
        display_img = image.copy()
        
    cv2.imshow(window_name, display_img)
    cv2.waitKey(0)

#CARGA Y PRE-PROCESAMIENTO DE IMAGEN

img = cv2.imread('monedas/monedas.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #En escala de grises

plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show() #Mostramos imagen original

# DETECCIÓN DE BORDES Y PREPARACIÓN MORFOLÓGICA
# Aplicamos Blur y Canny, mostramos resultados con los bordes
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
edges = cv2.Canny(blurred, 30, 120)
show_image('Bordes Canny', edges)

# Operaciones Morfológicas 
# Dilatamos para conectar bordes cercanos
dilation_kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, dilation_kernel, iterations=2)
show_image('Bordes Dilatados', dilated_edges)

# Cerramos agresivamente para rellenar pequeños agujeros y cerrar contornos, para a futuro facilitar el contorno
closing_kernel = np.ones((30, 30), np.uint8)
closed_edges = cv2.morphologyEx(
    dilated_edges, cv2.MORPH_CLOSE, closing_kernel, iterations=1
)
show_image('Bordes Dilatados y Cerrados', closed_edges)


#BÚSQUEDA Y FILTRADO INICIAL DE CONTORNOS

#Usamos la función External para no buscar bordes internos de las monedas y solo usar contornos
contours, hierarchy = cv2.findContours(
    closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)
contour_img = img.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2) # Dibuja contornos en verde
show_image('Contornos Encontrados', contour_img)

#Filtrado de Contornos por Área - 
#PASO EVITABLE, Ya que la imagen contorneada, y la imagen contorneada post-filtrado son iguales
#Pero mantenemos la función y los parametros para que a futuro este codigo sea re-utilizable para otras fotos.
objects_contours = []
MIN_AREA_THRESHOLD = 5000
MAX_AREA_THRESHOLD = 999999

for cnt in contours:
    area = cv2.contourArea(cnt)
    
    if area > MIN_AREA_THRESHOLD and area < MAX_AREA_THRESHOLD:
        objects_contours.append(cnt)

# Dibujar contornos filtrados
filtered_contour_img = img.copy()
cv2.drawContours(filtered_contour_img, objects_contours, -1, (255, 0, 0), 2) # Dibuja en azul
show_image('Contornos Filtrados (Monedas y Dados)', filtered_contour_img)


#CLASIFICACIÓN
#Diferenciar Monedas de Dados por Forma (Circularidad y Relación de Aspecto)

monedas_contours = []
dado_contours = []
separation_display_img = img.copy()

print("\n--- Diferenciando Monedas de Dados por Forma ---")

# Parámetros de clasificación
CIRCULARITY_THRESHOLD_1 = 0.79
ASPECT_RATIO_MIN_1 = 0.9
ASPECT_RATIO_MAX_1 = 1.1
CIRCULARITY_THRESHOLD_2 = 0.75
AREA_THRESHOLD_2 = 70000

for i, cnt in enumerate(objects_contours):
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    circularity = 0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Obtener el rectángulo delimitador para la relación de aspecto
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    
    is_moneda = False
    
    #Regla 1: Monedas con alta circularidad y relación de aspecto cercana a 1
    if circularity > CIRCULARITY_THRESHOLD_1 and (aspect_ratio > ASPECT_RATIO_MIN_1 and aspect_ratio < ASPECT_RATIO_MAX_1): 
        is_moneda = True
    #Regla 2: Monedas con circularidad aceptable pero que son pequeñas (evita problema chicas con mal borde)
    elif circularity > CIRCULARITY_THRESHOLD_2 and area < AREA_THRESHOLD_2:
        is_moneda = True

    # Asignación y visualización
    if is_moneda:
        monedas_contours.append(cnt)
        # Dibujar un rectángulo alrededor de la moneda en amarillo
        cv2.rectangle(separation_display_img, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.putText(separation_display_img, f'Objeto {i}: Moneda', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 2)
        print(f"Objeto {i}: Área={area:.2f}, Circ={circularity:.3f}, AR={aspect_ratio:.3f}. Clasificado como Moneda.")
    else:
        dado_contours.append(cnt)
        # Dibujar un rectángulo alrededor del dado en magenta
        cv2.rectangle(separation_display_img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        cv2.putText(separation_display_img, f'Objeto {i}: Dado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 255), 2)
        print(f"Objeto {i}: Área={area:.2f}, Circ={circularity:.3f}, AR={aspect_ratio:.3f}. Clasificado como Dado.")

# Mostrar el resultado final de la separación
print(f"\nTotal de Monedas detectadas: {len(monedas_contours)}")
print(f"Total de Dados detectados: {len(dado_contours)}")
show_image('Separacion Monedas y Dados (por Forma)', separation_display_img)

#Clasificar Monedas por Tamaño (Ancho del Bounding Box = Diametro moneda aprox) 
# Inicializar contadores
count_10c = 0
count_1p = 0
count_50c = 0
moneda_classification_img = img.copy()

print("\n--- Clasificando Monedas por Ancho del Bounding Box ---")

# Umbrales de ancho (Mantenidos de los originales, ASUMIDOS como ajustados)
WIDTH_10C_MAX = 320  # Ancho máximo estimado para 10 centavos
WIDTH_1P_MAX = 350   # Ancho máximo estimado para 1 Peso (y mínimo para 50c)

for cnt in monedas_contours: # Iteramos solo sobre los contornos clasificados como monedas
    
    # Obtener el bounding box (x, y, w, h)
    x, y, w, h = cv2.boundingRect(cnt)
    
    moneda_value = ""
    color = (0, 0, 0) # Color inicial
    
    # Clasificación basada en el ancho (w)
    if w < WIDTH_10C_MAX:
        moneda_value = "10 Centavos"
        count_10c += 1
        color = (0, 255, 0) # Verde
    elif w < WIDTH_1P_MAX:
        moneda_value = "1 Peso"
        count_1p += 1
        color = (255, 0, 0) # Azul
    else: # w >= WIDTH_1P_MAX
        moneda_value = "50 Centavos"
        count_50c += 1
        color = (0, 0, 255) # Rojo

    # Dibujar el bounding box y el texto de clasificación
    text_label = f'{moneda_value} (W:{w})'
    cv2.rectangle(moneda_classification_img, (x, y), (x+w, y+h), color, 2)
    cv2.putText(moneda_classification_img, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 2)
    
    print(f"Moneda clasificada: {moneda_value} (Ancho: {w})")

# Mostrar el resultado del conteo y la clasificación en consola
print("\n--- Conteo Final de Monedas ---")
print(f"Total 10 Centavos: {count_10c}")
print(f"Total 1 Peso: {count_1p}")
print(f"Total 50 Centavos: {count_50c}")

show_image('Clasificacion y Conteo de Monedas', moneda_classification_img)

# DETERMINACIÓN DEL VALOR DE LA CARA SUPERIOR DE LOS DADOS

print("\n--- Conteo de Puntos en los Dados (versión simplificada) ---")

final_dado_count_img = img.copy()
total_dado_value = 0
window_base = ' DADO '

#Recibimos roi en gris y en color para análisis y visualización
def contar_puntos_en_dado(roi_gray, roi_color, nombre_analisis, analisis=True):
    # 1) Blur y Umbralización Automática
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    _, bin_inv = cv2.threshold(
        blur, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    #Apertura para quitar ruido pequeño (Erosión y dilatación)
    kernel_open = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(bin_inv, cv2.MORPH_OPEN, kernel_open, iterations=1)
    if analisis:
        show_image(f'{nombre_analisis}: 2a - APERTURA Morfológica', clean)
        
    #Clausura para rellenar agujeros en los puntos (Dilatación y erosión)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    if analisis:
        show_image(f'{nombre_analisis}: 2b - CLAUSURA Morfológica', clean)

    #Contornos de los puntos
    # Usamos RETR_TREE para asegurar la detección de todos los contornos internos/externos (a diferencia de en monedas que usamos EXTERNAL)
    contours, _ = cv2.findContours(
        clean, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    punto_count = 0
    viz = roi_color.copy()

    # Áreas de filtrado de puntos
    roi_area = roi_gray.shape[0] * roi_gray.shape[1]
    min_area = roi_area * 0.01    # (Las areas de las caras de los costados, suelen ser la mitad o menos que las del frente)
    max_area = roi_area * 0.3    
    CIRCULARITY_MIN_punto = 0.4
    
    print(f"  Área ROI Dado: {roi_area}, min_area: {min_area}, max_area: {max_area}")
    print(f"\n--- analisis {nombre_analisis} ---")
    
    for c in contours:
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)
        
        print(f"  Contorno detectado: área={area:.1f}")

        # Filtrado por área
        if area < min_area or area > max_area:
            continue
        
        # Filtrado por circularidad
        circularity = 0
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter * perimeter)
        
        if circularity < CIRCULARITY_MIN_punto:
            continue

        # Si pasa los filtros, es un punto
        punto_count += 1
        x, y, w, h = cv2.boundingRect(c)
        
        cv2.rectangle(viz, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(viz, f'{punto_count}', (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"  Punto {punto_count}: área={area:.1f}, circ={circularity:.2f}")

    if analisis:
        show_image(f'{nombre_analisis}: 3 - Puntos Detectados (Valor {punto_count})', viz)

    return punto_count, viz


# LOOP PRINCIPAL SOBRE DADOS DETECTADOS
for i, cnt in enumerate(dado_contours):

    #Definición del ROI del dado con margen
    x, y, w, h = cv2.boundingRect(cnt)
    margin = 10

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(img.shape[1], x + w + margin)
    y2 = min(img.shape[0], y + h + margin)

    roi_color = img[y1:y2, x1:x2].copy()
    roi_gray  = gray[y1:y2, x1:x2]

    show_image(f'{window_base}{i}: 0 - ROI Original', roi_color)

    #Contar puntos en este dado
    nombre_analisis = f'{window_base}{i}'
    punto_count, roi_viz = contar_puntos_en_dado(
        roi_gray, roi_color, nombre_analisis, analisis=True
    )

    total_dado_value += punto_count

    #Pegar el ROI anotado de vuelta en la imagen final
    final_dado_count_img[y1:y2, x1:x2] = roi_viz

    #Dibujar bounding box y texto
    color_box = (255, 0, 255)
    cv2.rectangle(final_dado_count_img, (x, y), (x + w, y + h), color_box, 2)
    cv2.putText(final_dado_count_img, f'Dado: Valor {punto_count}',
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_box, 2)

    print(f"Dado {i}: {punto_count} puntos detectados.")


print(f"\nTotal del valor de todos los dados: {total_dado_value}")
print(f"Total de dados contados: {len(dado_contours)}")

show_image('Conteo de Puntos en Dados', final_dado_count_img)

cv2.destroyAllWindows()