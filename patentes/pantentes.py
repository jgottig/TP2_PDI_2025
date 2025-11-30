import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob # Necesario para buscar múltiples archivos

#Definimos funciones necesarias para mostrar los pasos de forma ordenada
def mostrar_paso(titulo, imagen, cmap='gray'):
    """Muestra una imagen con un título."""
    plt.figure(figsize=(10, 6))
    plt.title(titulo)
    if len(imagen.shape) == 3:
        plt.imshow(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    else:
        plt.imshow(imagen, cmap=cmap)
    plt.axis('off')
    plt.show()

def procesar_imagen(file_path):
    print(f"PROCESANDO IMAGEN: {file_path}")
    
    #CARGA DE IMAGENes
    img = cv2.imread(file_path)
    if img is None:
        print(f"No se pudo cargar la imagen {file_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # DETECCIÓN DE PLACA 
    # PREPROCESAMOS LA IMAGEN
    kernel_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 10)) 
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_hat) # Realce con Blackhat 
    scale_abs = cv2.convertScaleAbs(blackhat, alpha=4.5, beta=-550) # Contrastamos fuerte, luego al invertir las placas quedan casi bordeadas.
    mostrar_paso(f"1. Realce (BlackHat) - {file_path}", scale_abs)

    # UMBRALAMOS LA IMAGEN PARA BINARIZAR, USAMOS TECNICA DE INVERSIÓN
    _, th = cv2.threshold(scale_abs, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th_inv = 255 - th
    mostrar_paso(f"2.1 Threshold (Otsu invertido) - {file_path}", th_inv)

    # DETECCIÓN DE BORDES CON CANNY
    edges = cv2.Canny(th_inv, 50, 150)
    mostrar_paso(f"3.1 Bordes (Canny) - {file_path}", edges)

    # DILATACIÓN/CIERRE PARA UNIR BORDES DE LA PLACA
    kernel_patente = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)) 
    edges_dil = cv2.dilate(edges, kernel_patente, iterations=1)
    mostrar_paso(f"3.2 Bordes dilatados (para unir la placa) - {file_path}", edges_dil)

    closing_kernel = np.ones((5,5),np.uint8) 
    closed_edges = cv2.morphologyEx(edges_dil, cv2.MORPH_CLOSE, closing_kernel, iterations=1)
    mostrar_paso(f'Bordes Dilatados y Cerrados - {file_path}', closed_edges)

    # CONTORNOS CANDIDATOS A PLACA
    contours, _ = cv2.findContours(edges_dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_placas = img.copy()
    candidatos_placa = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        area = w * h

        if 2.0 < ar < 4.0 and 800 < area < 5000:
            roi = edges_dil[y:y+h, x:x+w]
            white_density = cv2.countNonZero(roi) / (w*h)

            if white_density < 0.5:
                continue

            candidatos_placa.append((x, y, w, h))
            cv2.rectangle(img_placas, (x, y), (x+w, y+h), (0, 255, 0), 2)

    mostrar_paso(f"3.3 Candidatos a placa por Canny - {file_path}", img_placas, cmap=None)

    # ELEGIR LA MEJOR PLACA Y RECORTARLA
    roi_patente = None
    if candidatos_placa:
        x_placa, y_placa, w_placa, h_placa = max(candidatos_placa, key=lambda r: r[2] * r[3])
        roi_patente = gray[y_placa:y_placa+h_placa, x_placa:x_placa+w_placa]
        mostrar_paso(f"4. ROI de la placa detectada - {file_path}", roi_patente)
    else:
        print("ADVERTENCIA: No se detectó ninguna placa candidata.")
        return

    #DETECCIÓN DE CARACTERES
    #PREPROCESAMIENTO DE LA ROI PARA CARACTERES
    roi_h, roi_w = roi_patente.shape[:2]
    
    # Suavizado
    blurred_roi = cv2.GaussianBlur(roi_patente, (1, 1), 0)
    mostrar_paso(f"5.1 ROI Suavizada (GaussianBlur) - {file_path}", blurred_roi) 
    
    #Umbral Otsu NORMAL: Caracteres = BLANCO, Fondo = NEGRO
    _, th_char = cv2.threshold(blurred_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mostrar_paso(f"5.2 ROI Umbralizada (Otsu NORMAL) - {file_path}", th_char)
    
    #Morfología para Limpieza y Relleno (afinar)
    kernel_close_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    closed_v_char = cv2.morphologyEx(th_char, cv2.MORPH_CLOSE, kernel_close_v, iterations=0) # iteraciones 0 = no hace nada
    
    kernel_thin = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    eroded = cv2.erode(closed_v_char, kernel_thin, iterations=1)
    final_char_img = cv2.subtract(closed_v_char, eroded) # Afinamiento: Resta la erosión al cierre

    mostrar_paso(f"5.4 Caracteres Finales (Thinning Suave y Limpieza) - {file_path}", final_char_img) 


    #ENCONTRAR Y FILTRAR CONTORNOS DE CARACTERES
    char_contours, _ = cv2.findContours(final_char_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # DEBUG: Mostrar todos los contornos antes de filtrar
    img_all_contours = cv2.cvtColor(final_char_img.copy(), cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_all_contours, char_contours, -1, (0, 0, 255), 1)
    mostrar_paso(f"5.5 Todos los Contornos Encontrados - {file_path}", img_all_contours, cmap=None)
    
    candidatos_caracter = []

    # Criterios de filtrado
    min_h = roi_h * 0.3
    max_h = roi_h * 0.65
    min_ar = 0.3           
    max_ar = 2.0           
    min_area = 10         
    max_area = roi_w * roi_h * 0.3

    for cc in char_contours:
        x_c, y_c, w_c, h_c = cv2.boundingRect(cc)
        ar = w_c / float(h_c)
        area = w_c * h_c

        if (min_h < h_c < max_h) and \
           (min_ar < ar < max_ar) and \
           (min_area < area < max_area):
            
            candidatos_caracter.append((x_c, y_c, w_c, h_c))

    #ORDENAR Y MOSTRAR LOS CARACTERES DETECTADOS
    roi_patente_color = cv2.cvtColor(roi_patente, cv2.COLOR_GRAY2BGR)
    candidatos_caracter.sort(key=lambda r: r[0]) 

    print(f"ROI H:{roi_h}, W:{roi_w}")
    print(f"Criterios: H({min_h:.1f}-{max_h:.1f}), AR({min_ar:.1f}-{max_ar:.1f}), Area({min_area}-{max_area:.0f})")
    print(f"Caracteres detectados: {len(candidatos_caracter)}")
    
    for i, (x_c, y_c, w_c, h_c) in enumerate(candidatos_caracter):
        cv2.rectangle(roi_patente_color, (x_c, y_c), (x_c+w_c, y_c+h_c), (0, 255, 0), 2)

    mostrar_paso(f"6. Caracteres detectados con Bounding Box (Final) - {file_path}", roi_patente_color, cmap=None)

    #RECORTAR Y MOSTRAR CARACTERES INDIVIDUALES
    if candidatos_caracter:
        num_chars = len(candidatos_caracter)
        plt.figure(figsize=(2 * num_chars, 4)) 

        #mostramos roi_patente, evitamos mostrar final_char
        for i, (x_c, y_c, w_c, h_c) in enumerate(candidatos_caracter):
            char_roi_bin = roi_patente[y_c:y_c+h_c, x_c:x_c+w_c]
            
            plt.subplot(1, num_chars, i + 1)
            plt.imshow(char_roi_bin, cmap='gray')
            plt.title(f"Char {i+1}")
            plt.axis('off')

        plt.suptitle(f"7. Caracteres Individuales Recortados (Ordenados) - {file_path}", y=0.95, fontsize=16)
        plt.show()


#ARMAMOS EJECUCIÓN PARA MÚLTIPLES IMÁGENES
# Buscar todos los archivos "img*.png" en la carpeta "patentes"
image_files = sorted(glob.glob('patentes/img*.png'))

if not image_files:
    print("No se encontraron archivos 'img*.png' en la carpeta 'patentes/'.")
else:
    for img_path in image_files:
        procesar_imagen(img_path)
    
print("\n--- ¡PROCESAMIENTO COMPLETO! ---")