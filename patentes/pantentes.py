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
    candidatos_caracter = []

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

    # AGREGAMOS ITERACIÓN DE UMBRALADO PARA OBTENER MAS PATENTES EN CASO DE NO DETECTAR.
    if not candidatos_placa:
        print(f"  [INFO] Otsu no detectó nada en {file_path}. Probando diferentes umbrales...")
        
        # Rango de umbrales a probar
        for t_val in range(50, 100, 15):
            _, th_iter = cv2.threshold(scale_abs, t_val, 255, cv2.THRESH_BINARY)
            th_inv_iter = 255 - th_iter
            
            # Mismos pasos de detección
            edges_iter = cv2.Canny(th_inv_iter, 50, 150)
            edges_dil_iter = cv2.dilate(edges_iter, kernel_patente, iterations=1)
            
            # Busco contornos
            contours_iter, _ = cv2.findContours(edges_dil_iter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            candidatos_iter = []
            for c in contours_iter:
                x, y, w, h = cv2.boundingRect(c)
                ar = w / float(h)
                area = w * h
                
                # Mismos filtros de tamaño/forma
                if 2.0 < ar < 4.0 and 800 < area < 5000:
                    roi_iter = edges_dil_iter[y:y+h, x:x+w]
                    white_density_iter = cv2.countNonZero(roi_iter) / (w*h)
                    
                    if white_density_iter >= 0.5:
                        candidatos_iter.append((x, y, w, h))
            
            # Si encontramos algo en esta iteración, guardamos y salimos
            if candidatos_iter:
                print(f"  [EXITO] Candidatos encontrados con umbral manual: {t_val}")
                candidatos_placa = candidatos_iter
                
                # Visualización del rescate para feedback
                img_rescue = img.copy()
                for (x, y, w, h) in candidatos_placa:
                    cv2.rectangle(img_rescue, (x, y), (x+w, y+h), (0, 0, 255), 2)
                mostrar_paso(f"3.3 (RESCATE) Candidatos con Umbral {t_val} - {file_path}", img_rescue, cmap=None)
                break

    # === RESCATE 2: Detección Directa de Caracteres (Adaptive check) ===
    if not candidatos_placa:
        PARAMETROS = {
            'bloque_umbral': 13, 
            'constante_umbral': 18,
            'ar_min': 0.4, 
            'ar_max': 3.5,
            'altura_min': 0.030, # Altura relativa img
            'altura_max': 0.100,
            'alineacion_y': 0.20, 
            'alineacion_h': 0.32, 
            'alineacion_dx': 1.6
        }
        
        h_img, w_img = gray.shape
        
        # Umbralado
        bs = PARAMETROS['bloque_umbral']
        if bs % 2 == 0: bs += 1
        thresh_char = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, bs, PARAMETROS['constante_umbral'])
        
        # Contornos
        cnts_char, _ = cv2.findContours(thresh_char, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        candidatos_raw = []
        for c in cnts_char:
            x, y, w, h = cv2.boundingRect(c)
            
            # Validación de dimensiones del posible caracter
            ar = float(h) / w
            if (PARAMETROS['ar_min'] < ar < PARAMETROS['ar_max']) and \
               (h_img * PARAMETROS['altura_min'] < h < h_img * PARAMETROS['altura_max']):
                candidatos_raw.append((x,y,w,h))
        
        # Agrupamiento
        candidatos_raw.sort(key=lambda r: r[0])
        grupos = []
        visitados = set()
        
        for i in range(len(candidatos_raw)):
            if i in visitados: continue
            grupo = [candidatos_raw[i]]
            visitados.add(i)
            ref = candidatos_raw[i]
            
            for j in range(i+1, len(candidatos_raw)):
                if j in visitados: continue
                cand = candidatos_raw[j]
                
                rx, ry, rw, rh = ref
                cx, cy, cw, ch = cand
                
                # Centros verticales
                cy_ref = ry + rh/2
                cy_cand = cy + ch/2
                
                # Alineación vertical y similitud de altura
                if abs(cy_ref - cy_cand) > rh * PARAMETROS['alineacion_y']: continue
                if abs(rh - ch) > rh * PARAMETROS['alineacion_h']: continue
                
                # Distancia horizontal
                dist_x = cx - (rx + rw)
                if dist_x > rw * PARAMETROS['alineacion_dx'] or dist_x < -rw * 0.5: continue
                
                grupo.append(cand)
                visitados.add(j)
                ref = cand
            
            if len(grupo) >= 3:
                grupos.append(grupo)
        
        mejor_grupo = []
        if grupos:
            mejor_grupo = max(grupos, key=len)
            
            # Filtrar si hay exceso de candidatos (>6), priorizando los de altura mediana
            if len(mejor_grupo) > 6:
                alturas = [r[3] for r in mejor_grupo]
                while len(mejor_grupo) > 6:
                    mediana = np.median([r[3] for r in mejor_grupo])
                    # Remover el que más se aleja de la mediana
                    mejor_grupo.sort(key=lambda r: abs(r[3]-mediana), reverse=True)
                    mejor_grupo.pop(0) 
            
            # Orden final por posición X
            mejor_grupo.sort(key=lambda r: r[0])

        if len(mejor_grupo) >= 6:
             print(f"  [EXITO] Rescate por caracteres detectó {len(mejor_grupo)}!")
             
             # Reconstruir ROI de patente basada en el grupo
             min_x = min(r[0] for r in mejor_grupo)
             min_y = min(r[1] for r in mejor_grupo)
             max_x = max(r[0]+r[2] for r in mejor_grupo)
             max_y = max(r[1]+r[3] for r in mejor_grupo)
             
             # Agregar un margen
             pad_x = int((max_x - min_x) * 0.05)
             pad_y = int((max_y - min_y) * 0.15)
             
             px = max(0, min_x - pad_x)
             py = max(0, min_y - pad_y)
             pw = (max_x + pad_x) - px
             ph = (max_y + pad_y) - py
             
             candidatos_placa = [(px, py, pw, ph)]
             
             # Guardar caracteres encontrados (coordenadas relativas a la ROI)
             candidatos_caracter = []
             for (cx, cy, cw, ch) in mejor_grupo:
                 candidatos_caracter.append((cx - px, cy - py, cw, ch))
                 
             mostrar_paso(f"3.3 (RESCATE CARACTERES) - {file_path}", img[py:py+ph, px:px+pw], cmap=None)

    # ELEGIR LA MEJOR PLACA Y RECORTARLA
    roi_patente = None
    if candidatos_placa:
        x_placa, y_placa, w_placa, h_placa = max(candidatos_placa, key=lambda r: r[2] * r[3])
        roi_patente = gray[y_placa:y_placa+h_placa, x_placa:x_placa+w_placa]
        mostrar_paso(f"4. ROI de la placa detectada - {file_path}", roi_patente)
    else:
        print("ADVERTENCIA: No se detectó ninguna placa candidata.")
        return

    #DETECCIÓN DE CARACTERES (Solo si no fueron detectados por el rescate previo)
    if not candidatos_caracter:
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
        
        # Criterios de filtrado - ITERACION
        configs_char = [
            {'min_h': 0.3, 'max_h': 0.55, 'min_ar': 0.3, 'max_ar': 2.0},  # Config actual (estricta)
            {'min_h': 0.25, 'max_h': 0.65, 'min_ar': 0.25, 'max_ar': 2.5}, # Laxa 1
            {'min_h': 0.2, 'max_h': 0.75, 'min_ar': 0.2, 'max_ar': 3.0},   # Laxa 2
            {'min_h': 0.15, 'max_h': 0.85, 'min_ar': 0.15, 'max_ar': 3.5}, # Laxa 3
            {'min_h': 0.1, 'max_h': 0.95, 'min_ar': 0.1, 'max_ar': 4.0}    # Muy laxa
        ]

        min_area = 10         
        max_area = roi_w * roi_h * 0.3 # Este lo dejamos fijo por ahora
        
        for cfg in configs_char:
            candidatos_temp = []
            
            # Umbrales actuales
            th_min_h = roi_h * cfg['min_h']
            th_max_h = roi_h * cfg['max_h']
            th_min_ar = cfg['min_ar']
            th_max_ar = cfg['max_ar']

            for cc in char_contours:
                x_c, y_c, w_c, h_c = cv2.boundingRect(cc)
                ar = w_c / float(h_c)
                area = w_c * h_c

                if (th_min_h < h_c < th_max_h) and \
                (th_min_ar < ar < th_max_ar) and \
                (min_area < area < max_area):
                    
                    candidatos_temp.append((x_c, y_c, w_c, h_c))
            
            # Si encontramos 6 o más, nos quedamos con esta config y salimos
            if len(candidatos_temp) >= 6:
                candidatos_caracter = candidatos_temp
                print(f"  [INFO] Se encontraron {len(candidatos_caracter)} caracteres con config: {cfg}")
                break
                
        # VALIDACION FINAL: Si no llegamos a 6, descartamos la imagen
        if len(candidatos_caracter) < 6:
            print(f"  [FALLO] No se lograron detectar 6 caracteres en {file_path} (Encontrados: {len(candidatos_caracter)}). Saltando visualización.")
            return

    #ORDENAR Y MOSTRAR LOS CARACTERES DETECTADOS
    roi_h, roi_w = roi_patente.shape[:2] # Recalcular por si venimos del rescate 2
    roi_patente_color = cv2.cvtColor(roi_patente, cv2.COLOR_GRAY2BGR)
    candidatos_caracter.sort(key=lambda r: r[0]) 

    print(f"ROI H:{roi_h}, W:{roi_w}")
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