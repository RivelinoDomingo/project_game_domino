import cv2
import numpy as np
import math
import argparse
from skimage.morphology import skeletonize
# from scipy.signal import find_peaks


parser = argparse.ArgumentParser(description='Processa imagens')
parser.add_argument('imagem', help='Caminho para o arquivo de imagem')
parser.add_argument('-z', '--zoom', type=float, help='Valor float para nivel de zoom da imagem, padrão 0.8')
parser.add_argument('-p', '--proximidade', type=int, help='Distancia minima entre as pedras padrão 30')
parser.add_argument('-d', '--debug', action='store_true', help='Ativa o modo depuração para melhor análise' )
args = parser.parse_args()

zoom_factor = args.zoom
DISTANCIA_MINIMA = args.proximidade

# No início do arquivo, adicione estes parâmetros para fácil ajuste:
CONFIG_VALES = {
    'angulo_min': 10,      # Ângulo mínimo para considerar V
    'angulo_max': 96,      # Ângulo máximo (você já ajustou)
    'profundidade_defect': 2000,  # Profundidade mínima do convexity defect
    'distancia_filtro': 14,  # Distância para filtrar pontos próximos
    'distancia_corte': 60,   # Distância máxima para linha de corte
    'area_min': 1000,        # Área minima dos contornos das pedras
}

if args.proximidade is None:
    DISTANCIA_MINIMA = 37

debug_mode = args.debug

if zoom_factor is None:
        zoom_factor = 1.0

if debug_mode:
    print("MODO DEBUG ATIVADO - Saida Turbinada")

def pipeline_blackhat(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return
    # Área de zoom
    img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   # ====================================================================
    # 1. ENCONTRAR A SILHUETA SÓLIDA BASE
    # ====================================================================
    _, mask_branca = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours_externos, _ = cv2.findContours(mask_branca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_pedra_solida = np.zeros_like(mask_branca)
    cv2.drawContours(mask_pedra_solida, contours_externos, -1, 255, thickness=cv2.FILLED)

    # ====================================================================
    # 2. O GOLPE DE ESPADA (The Blackhat Cleaver)
    # ====================================================================
    # O Blackhat acha as fendas escuras que separam as pedras paralelas
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

    # cv2.imshow('1 - MaskBranca', mask_branca)
    # cv2.imshow('1 - Mask Pedra', mask_pedra_solida)
    # cv2.imshow('1 - BlackHat', blackhat)

    # Usamos 80 (igual ao seu código) para pegar as fendas
    _, mask_fendas = cv2.threshold(blackhat, 80, 255, cv2.THRESH_BINARY)

    # Engrossamos a fenda só um pouquinho para garantir que ela rache a máscara branca
    kernel_fenda = np.ones((2,2), np.uint8)
    mask_fendas_grossas = cv2.dilate(mask_fendas, kernel_fenda, iterations=1)
    # cv2.imshow("1 - Mascara Fendas", mask_fendas_grossas)

    # ⚔️ FATIAMOS A MÁSCARA! A fenda rasga a pedra, e o traço faz um furo no centro.
    mask_pedra_solida[mask_fendas_grossas == 255] = 0

    # ====================================================================
    # 3. O CURATIVO TOPOLÓGICO
    # ====================================================================
    # O RETR_EXTERNAL é mágico: ele ignora os furos no meio do dominó (traço),
    # mas respeita o corte da fenda que dividiu a borda externa!
    contours_pre_melt, _ = cv2.findContours(mask_pedra_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Redesenhamos tudo. Os furos internos somem, a pedra fica forte, e a fenda é mantida!
    mask_pedra_solida = np.zeros_like(mask_branca)
    cv2.drawContours(mask_pedra_solida, contours_pre_melt, -1, 255, thickness=cv2.FILLED)
    # cv2.imshow('1 - Pedra Solida Final', mask_pedra_solida)
    # Agora sim, aplicar erosão mais suave
    kernel_derreter = np.ones((2, 2), np.uint8)  # Kernel menor agora
    mask_derretida = cv2.erode(mask_pedra_solida, kernel_derreter, iterations=2)
    # mask_derretida = mask_pedra_solida
    # cv2.imshow('2 - Pedra Solida Final', mask_derretida)

    # Limpeza da mascara
    # 1. Encontrar os contornos
    contornos, _ = cv2.findContours(mask_derretida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.imshow('2 - Mask derretida', mask_derretida)

    # 2. Criar uma máscara preta (do mesmo tamanho da original)
    altura, largura = mask_derretida.shape[:2]
    mask_filtrada = np.zeros((altura, largura), dtype=np.uint8)

    # 3. Filtrar e desenhar apenas os contornos com área > 800
    area_minima = 1500

    for cnt in contornos:
        if cv2.contourArea(cnt) > area_minima:
            cv2.drawContours(mask_filtrada, [cnt], -1, 255, thickness=cv2.FILLED)
            # thickness=cv2.FILLED preenche o contorno completamente (recomendado para máscara)
    # cv2.imshow('3 - Mask Filtrada', mask_filtrada)

    # Substitua a chamada anterior por:
    pontos_vale = detectar_vales_focado(mask_filtrada, gray)
    # img_debug = img.copy()

    if len(pontos_vale) > 0:
        if debug_mode:
            img_debug_final = visualizar_vales_detalhado(img, mask_pedra_solida, pontos_vale)

        # IMPORTANTE: Encontrar o contorno principal da máscara ATUAL
        contours_atual, _ = cv2.findContours(mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours_atual:
            contorno_principal = max(contours_atual, key=cv2.contourArea)

            # Agora sim, encontrar pares de corte usando o contorno correto
            raio_corte = CONFIG_VALES['distancia_corte']  # Ajuste conforme necessário
            pares_corte = encontrar_pares_corte(pontos_vale, contorno_principal, raio_corte)

            # Aplicar cortes inteligentes
            contours_depois = cortar_nos_vales_inteligente(mask_filtrada, pontos_vale, raio_corte)
            # cv2.imshow("4 - FINAL", mask_pedra_cortada)
            # contours_depois = cv2.findContours(mask_pedra_cortada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # for cnt in contours_depois:
            #     area = cv2.contourArea(cnt)
            #     if area > 800:
            #         cv2.drawContours(img_debug, [cnt], -1, (255, 0, 0), 2)          # contorno azul
            #         cv2.imshow('1 - Debug', img_debug)
            #         cv2.waitKey(0)

            # Visualizar resultado dos cortes
            if debug_mode:
                img_cortes = visualizar_cortes(img_debug_final, mask_derretida, mask_derretida, contours_depois, pares_corte, "2 - Cortes Aplicados")

            # Usar a máscara cortada para o resto do pipeline
            # mask_pedra_solida = mask_pedra_cortada
            # cv2.imshow('3 - Máscara Final Cortada', mask_pedra_solida)
        else:
            print("⚠️ Nenhum contorno encontrado na máscara!")







    # contours, _ = cv2.findContours(mask_branca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    candidatos = []
    out = img.copy()

    if debug_mode:
        print(f"Contornos encontrados: {len(contours_depois)}")

    for cnt in contours_depois:
        area = cv2.contourArea(cnt)
        # if area < 10 or area > 2200:
        #     continue
        if   4500 > area > 500:
            rect = cv2.minAreaRect(cnt)
            center, size, angle = rect
            (cx, cy), (w_box, h_box), angle = rect

            if w_box == 0 or h_box == 0:
                continue

            width, height = size

            ratio = width/height

            margem = 1.1

            rect_pedra = (center, (width*margem, height*margem), angle)

            candidatos.append({
                'rect_pedra': rect_pedra,
                'rect_traco': rect,
                'centro': center
            })
    if debug_mode:
        print(f"Candidatos aprovados: {len(candidatos)}")



    pedras_unicas = []
    global DISTANCIA_MINIMA

    for cand in candidatos:
        cx1, cy1 = cand['centro']
        duplicata = False

        for p in pedras_unicas:
            cx2, cy2 = p['centro']
            if math.hypot(cx2 - cx1, cy2 - cy1) < DISTANCIA_MINIMA:
                duplicata = True
                break

        if not duplicata:
            pedras_unicas.append(cand)


    # ==========================================
    # PASSO 2: O Filtro da "Área de Influência" (O Maior Bando)
    # ==========================================
    DISTANCIA_CONEXAO = 100  # Tamanho da "Área de influência" de cada pedra

    visitados = set()
    todos_os_bandos = []

    for i, p1 in enumerate(pedras_unicas):
        # Se essa pedra já entrou num bando antes, ignoramos
        if i in visitados:
            continue

        # Começamos um novo bando com essa pedra
        bando_atual = [p1]
        visitados.add(i)

        # A "Fila de Expansão" (vai checar os amigos dos amigos)
        fila_de_expansao = [p1]

        while fila_de_expansao:
            pedra_foco = fila_de_expansao.pop(0)
            cx_foco, cy_foco = pedra_foco['centro']

            # Procura novos amigos para puxar para o bando
            for j, p2 in enumerate(pedras_unicas):
                if j not in visitados:
                    cx2, cy2 = p2['centro']
                    dist = math.hypot(cx2 - cx_foco, cy2 - cy_foco)

                    # Se a pedra está dentro da área de influência, entra pro bando!
                    if dist <= DISTANCIA_CONEXAO:
                        bando_atual.append(p2)
                        visitados.add(j)
                        # Coloca ela na fila para a área de influência dela também ser checada!
                        fila_de_expansao.append(p2)

        # Guarda o bando que acabamos de formar
        todos_os_bandos.append(bando_atual)

    # ==========================================
    # PASSO 3: Sobrevivência do Mais Forte
    # ==========================================
    if todos_os_bandos:
        # A função max() com 'key=len' pega automaticamente a lista que tem mais itens!
        maior_bando = max(todos_os_bandos, key=len)
        pedras_aprovadas = maior_bando
    else:
        pedras_aprovadas = []

    print(f"Pedras encontradas: {len(pedras_aprovadas)}")

    # Ordena as pedras de cima para baixo (pelo eixo Y do centro)
    pedras_aprovadas.sort(key=lambda x: x['centro'][1])

    for d in pedras_aprovadas:

        # Enviamos a imagem original limpa (img) e o retângulo da pedra
        pts_cima, pts_baixo = extrair_e_contar(img, d['rect_pedra'])
        texto = f"{pts_cima}|{pts_baixo}"
        # box_traco = np.int32(cv2.boxPoints(d['rect_traco']))
        # cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 2)

        box_pedra = np.int32(cv2.boxPoints(d['rect_pedra']))
        cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 2)
        # Escrever o resultado na imagem, do lado da pedra!

        cx, cy = int(d['centro'][0]), int(d['centro'][1])

        # Fundo preto para o texto ficar legível
        cv2.putText(out, texto, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Texto em branco
        cv2.putText(out, texto, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"Pedra encontrada: {texto}")

    print(f"Pedras aprovadas: {len(pedras_aprovadas)}")
    # cv2.imshow("Pedra Solida", mask_pedra_solida)
    # cv2.imshow("Mask Branca", mask_branca)
    # cv2.imshow("Mask Soldada", mask_soldada)
    cv2.imshow("Resultado Final Limpo", out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def ordenar_pontos(pts):
    # Inicializa uma lista de coordenadas que serão ordenadas
    # [top-left, top-right, bottom-right, bottom-left]
    rect = np.zeros((4, 2), dtype="float32")

    # O ponto superior-esquerdo terá a menor soma, o inferior-direito a maior
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # O ponto superior-direito terá a menor diferença, o inferior-esquerdo a maior
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def extrair_e_contar(img, rect_pedra):
    # Pega as 4 quinas da caixa verde
    box = cv2.boxPoints(rect_pedra)
    pts = ordenar_pontos(box)

    # Forçar a imagem a ficar sempre "em pé" (altura maior que largura)
    dist_0_1 = np.linalg.norm(pts[0] - pts[1]) # Largura superior
    dist_0_3 = np.linalg.norm(pts[0] - pts[3]) # Altura esquerda

    if dist_0_1 > dist_0_3:
        # Se estiver deitada, nós rotacionamos os pontos em 90 graus
        pts = np.array([pts[1], pts[2], pts[3], pts[0]], dtype="float32")

    # Criamos o "molde" perfeito de 40x80 pixels
    dst = np.array([
        [0, 0],
        [39, 0],
        [39, 79],
        [0, 79]
    ], dtype="float32")

    # Mágica: Estica a pedra inclinada da foto para caber no nosso molde perfeito
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (40, 80))

    # Guilhotina: Corta o molde ao meio
    metade_cima = warped[0:40, 0:40]
    metade_baixo = warped[40:80, 0:40]

    def contar_bolinhas(metade):
            gray = cv2.cvtColor(metade, cv2.COLOR_BGR2GRAY)

            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # BLINDAGEM 1: Apagar as bordas da imagem (3 pixels)
            # Isso impede que a sombra da beirada do dominó seja contada como bolinha
            h, w = thresh.shape
            cv2.rectangle(thresh, (0, 0), (w, h), 0, 3)

            kernel = np.ones((2,2), np.uint8)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

            contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pontos = 0

            for c in contornos:
                area = cv2.contourArea(c)

                # A sua área super calibrada pelo GIMP! (Dei uma margem de segurança 25 a 85)
                if 32 < area < 85:
                    # BLINDAGEM 2: O filtro de formato (Circularidade)
                    perimetro = cv2.arcLength(c, True)
                    if perimetro == 0: continue

                    circularidade = 4 * np.pi * (area / (perimetro * perimetro))

                    # Se for redondo o suficiente (Círculo = 1.0, Quadrado ~0.78)
                    if circularidade > 0.2:
                        pontos += 1

            # BLINDAGEM 3: Trava matemática máxima de um dominó
            return min(pontos, 6)

    pts_cima = contar_bolinhas(metade_cima)
    pts_baixo = contar_bolinhas(metade_baixo)

    # Opcional: mostrar as pedras extraídas para você ver a mágica acontecendo (comente depois)
    # cv2.imshow("Pedra Extraida", warped)
    # cv2.waitKey(0)

    return pts_cima, pts_baixo


'''
def detectar_vales_em_v(mask_pedras, gray_img):
    """
    Detecta os vales em V entre pedras de dominó encostadas
    """
    # 1. Encontrar o esqueleto da máscara para achar a linha central
    skeleton = skeletonize(mask_pedras // 255).astype(np.uint8) * 255

    # 2. Detectar cantos (os V's) usando Harris Corner Detection
    corners = cv2.cornerHarris(np.float32(mask_pedras), blockSize=5, ksize=3, k=0.04)

    # Dilatar os cantos para pegar bem os pontos de estrangulamento
    corners = cv2.dilate(corners, None)

    # Threshold para pegar apenas os cantos mais fortes (os V's entre pedras)
    corner_threshold = 0.01 * corners.max()
    corner_mask = (corners > corner_threshold).astype(np.uint8) * 255

    # 3. Encontrar os pontos de mínimo (estrangulamentos) no contorno
    contours, _ = cv2.findContours(mask_pedras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    pontos_corte = []

    if contours:
        # Pegar o maior contorno (grupo de pedras)
        main_contour = max(contours, key=cv2.contourArea)

        # Suavizar o contorno para reduzir ruído
        epsilon = 0.001 * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        # Analisar curvatura ao longo do contorno
        for i in range(len(approx)):
            # Pegar 3 pontos consecutivos
            p1 = approx[i][0]
            p2 = approx[(i + 1) % len(approx)][0]
            p3 = approx[(i + 2) % len(approx)][0]

            # Calcular ângulo entre os segmentos
            v1 = p1 - p2
            v2 = p3 - p2

            # Normalizar vetores
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)

            if n1 > 0 and n2 > 0:
                v1_norm = v1 / n1
                v2_norm = v2 / n2

                # Produto escalar para achar o ângulo
                cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
                angle = np.arccos(cos_angle) * 180 / np.pi

                # Ângulos agudos (< 120°) indicam vales em V
                if 10 < angle < 120:
                    # Verificar se é um ponto de estrangulamento (distância para o outro lado)
                    pontos_corte.append(p2)

    return np.array(pontos_corte)

def cortar_nos_vales(mask_pedra_solida, pontos_corte):
    """
    Aplica cortes nos pontos de vale detectados
    """
    if len(pontos_corte) < 2:
        return mask_pedra_solida

    mask_cortada = mask_pedra_solida.copy()

    # Para cada par de pontos de vale opostos
    for i in range(0, len(pontos_corte), 2):
        if i + 1 < len(pontos_corte):
            p1 = tuple(pontos_corte[i])
            p2 = tuple(pontos_corte[i+1])

            # Verificar se os pontos estão em lados opostos
            dist = np.linalg.norm(np.array(p1) - np.array(p2))
            if dist < CONFIG_VALES['distancia_corte']:  # Threshold de distância para vales opostos
                # Desenhar linha de corte
                cv2.line(mask_cortada, p1, p2, 0, thickness=3)

    return mask_cortada

def refinar_separacao_pedras(mask_fatiada_final, gray_img):
    """
    Versão melhorada da separação de pedras usando detecção de vales
    """
    # 1. Encontrar esqueleto para ter a linha média
    skeleton = skeletonize(mask_fatiada_final // 255).astype(np.uint8) * 255

    # 2. Transformada de distância para achar os estrangulamentos
    dist_transform = cv2.distanceTransform(mask_fatiada_final, cv2.DIST_L2, 5)

    # Normalizar
    dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

    # Encontrar mínimos locais (onde as pedras se tocam)
    kernel = np.ones((15, 15), np.uint8)
    local_min = cv2.erode(dist_transform, kernel)
    local_min = (dist_transform == local_min) & (dist_transform < 0.3)

    # 3. Watershed para separação precisa
    # Preparar marcadores
    _, markers = cv2.connectedComponents((dist_transform > 0.5).astype(np.uint8))
    markers = markers + 1
    markers[local_min] = 0

    # Aplicar watershed
    img_color = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    cv2.watershed(img_color, markers)

    # Criar máscara separada
    mask_separada = np.zeros_like(mask_fatiada_final)
    mask_separada[markers > 1] = 255

    # 4. Remover linhas de watershed muito grossas com morfologia
    kernel_clean = np.ones((3, 3), np.uint8)
    mask_separada = cv2.morphologyEx(mask_separada, cv2.MORPH_CLOSE, kernel_clean)

    return mask_separada
'''
'''
def detectar_vales_curvatura_avancado(mask_pedras, gray_img):
    """
    Versão otimizada do método de curvatura com múltiplas estratégias
    """
    contours, _ = cv2.findContours(mask_pedras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.array([])

    main_contour = max(contours, key=cv2.contourArea)
    pontos_vale = []

    # Estratégia 1: Diferentes níveis de suavização do contorno
    for suavizacao in [0.001, 0.002, 0.003, 0.005]:
        epsilon = suavizacao * cv2.arcLength(main_contour, True)
        approx = cv2.approxPolyDP(main_contour, epsilon, True)

        # Estratégia 2: Diferentes tamanhos de janela para análise de curvatura
        for janela in [2, 3, 4, 5]:
            for i in range(len(approx)):
                # Pegar pontos com espaçamento variável
                idx_antes = (i - janela) % len(approx)
                idx_depois = (i + janela) % len(approx)

                p1 = approx[idx_antes][0]
                p2 = approx[i][0]
                p3 = approx[idx_depois][0]

                # Calcular vetores
                v1 = p1 - p2
                v2 = p3 - p2

                # Normalizar
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)

                if n1 > 0 and n2 > 0:
                    # Ângulo entre vetores
                    cos_angle = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
                    angle = np.arccos(cos_angle) * 180 / np.pi

                    # Critério mais flexível para vales
                    if 20 < angle < 100:
                        # Verificar se é um ponto de mínimo local
                        # (a distância para o centro é menor que a média)
                        dist_centro = np.linalg.norm(p2 - main_contour.mean(axis=0)[0])

                        # Verificar vizinhança para confirmar que é um vale
                        vizinhos = []
                        for offset in [-3, -2, -1, 1, 2, 3]:
                            idx_viz = (i + offset) % len(approx)
                            vizinhos.append(approx[idx_viz][0])

                        dist_media_vizinhos = np.mean([np.linalg.norm(v - main_contour.mean(axis=0)[0])
                                                       for v in vizinhos])

                        # Se é um mínimo local (mais próximo do centro que vizinhos)
                        if dist_centro < dist_media_vizinhos * 0.95:
                            pontos_vale.append(p2)

    # Remover duplicatas
    return filtrar_pontos_inteligente(np.array(pontos_vale))

def detectar_vales_gradiente_intensidade(mask_pedras, gray_img):
    """
    Detecta vales analisando o gradiente de intensidade ao longo da borda
    """
    # Criar uma máscara da borda
    mask_borda = cv2.Canny(mask_pedras, 50, 150)

    # Dilatar para pegar uma região de busca
    kernel = np.ones((3,3), np.uint8)
    mask_borda = cv2.dilate(mask_borda, kernel, iterations=2)

    # Encontrar pontos onde a intensidade muda bruscamente (vales)
    # Usar Sobel para detectar bordas fortes
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)

    # Combinar com a máscara da borda
    magnitude[mask_borda == 0] = 0

    # Encontrar picos de magnitude (onde há mudança brusca = vales)
    from scipy.ndimage import maximum_filter
    local_max = maximum_filter(magnitude, size=10) == magnitude
    pontos_alto_gradiente = np.where((magnitude > 50) & local_max)

    pontos_vale = []
    for y, x in zip(pontos_alto_gradiente[0], pontos_alto_gradiente[1]):
        # Verificar se está próximo do contorno
        if mask_pedras[y, x] > 0:
            # Verificar se é um ponto de curvatura alta
            pontos_vale.append([x, y])

    return np.array(pontos_vale)
'''

def detectar_vales_multi_escala_otimizado(mask_pedras, gray_img):
    """
    Versão otimizada do multi-escala com mais níveis e melhor filtragem
    """
    todos_pontos = []

    # Mais escalas para melhor cobertura
    # escalas = [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]
    escalas = [0.95, 0.97, 1.0, 1.02, 1.05]

    h, w = mask_pedras.shape

    for escala in escalas:
        # Redimensionar
        nova_h, nova_w = int(h * escala), int(w * escala)
        mask_scaled = cv2.resize(mask_pedras, (nova_w, nova_h), interpolation=cv2.INTER_LINEAR)

        # Aplicar blur para suavizar em escalas menores
        if escala < 0.9:
            mask_scaled = cv2.GaussianBlur(mask_scaled, (5, 5), 0)

        # Detectar vales em múltiplas orientações
        for angulo in [0, 5, 10, 15, 20]:
        # for angulo in [0]:
            # Rotacionar a máscara
            if angulo != 0:
                M = cv2.getRotationMatrix2D((nova_w/2, nova_h/2), angulo, 1.0)
                mask_rotated = cv2.warpAffine(mask_scaled, M, (nova_w, nova_h))
            else:
                mask_rotated = mask_scaled

            # Detectar vales na versão rotacionada
            pontos = detectar_vales_em_v_simples(mask_rotated)

            if len(pontos) > 0:
                # Rotacionar de volta se necessário
                if angulo != 0:
                    M_inv = cv2.getRotationMatrix2D((nova_w/2, nova_h/2), -angulo, 1.0)
                    for i in range(len(pontos)):
                        p = np.array([pontos[i][0], pontos[i][1], 1])
                        p_back = M_inv @ p
                        pontos[i] = p_back[:2]

                # Escalar coordenadas de volta
                pontos = (pontos / escala).astype(np.int32)
                todos_pontos.extend(pontos)

    # Filtrar e agrupar pontos próximos
    return agrupar_pontos_proximos(np.array(todos_pontos), CONFIG_VALES['distancia_filtro'])   # Como só preciso dos pontos brutos não preciso disso, é voltado para exibição
    # return todos_pontos

def detectar_vales_em_v_simples(mask_pedras):
    """
    Versão simplificada e rápida para usar no multi-escala
    """
    contours, _ = cv2.findContours(mask_pedras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([])

    main_contour = max(contours, key=cv2.contourArea)
    pontos_vale = []

    # Análise mais agressiva de curvatura
    for i in range(len(main_contour)):
        p1 = main_contour[i-2][0] if i >= 2 else main_contour[0][0]
        p2 = main_contour[i][0]
        p3 = main_contour[(i+2) % len(main_contour)][0]

        v1 = p1 - p2
        v2 = p3 - p2

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 > 0 and n2 > 0:
            cos_angle = np.clip(np.dot(v1/n1, v2/n2), -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi

            # Threshold mais baixo para pegar mais vales
            if CONFIG_VALES['angulo_min'] < angle < CONFIG_VALES['angulo_max']:
                pontos_vale.append(p2)

    return np.array(pontos_vale)


def agrupar_pontos_proximos(pontos, raio=15):
    """
    Agrupa pontos próximos e retorna o centróide de cada grupo
    """
    if len(pontos) == 0:
        return pontos

    grupos = []
    pontos_usados = set()

    for i, p1 in enumerate(pontos):
        if i in pontos_usados:
            continue

        grupo = [p1]
        pontos_usados.add(i)

        for j, p2 in enumerate(pontos):
            if j not in pontos_usados:
                dist = np.linalg.norm(p1 - p2)
                if dist < raio:
                    grupo.append(p2)
                    pontos_usados.add(j)

        grupos.append(grupo)

    # Calcular centróide de cada grupo
    pontos_finais = []
    for grupo in grupos:
        centroide = np.mean(grupo, axis=0).astype(np.int32)
        pontos_finais.append(centroide)

    return np.array(pontos_finais)

def filtrar_pontos_inteligente(pontos, distancia_min=10):
    """
    Filtra pontos duplicados mas mantém aqueles em regiões diferentes
    """
    if len(pontos) == 0:
        return pontos

    # Ordenar por confiança (ângulo mais próximo de 60° é melhor)
    pontos_filtrados = []

    for p1 in pontos:
        duplicado = False
        for p2 in pontos_filtrados:
            dist = np.linalg.norm(p1 - p2)
            if dist < distancia_min:
                duplicado = True
                break
        if not duplicado:
            pontos_filtrados.append(p1)

    return np.array(pontos_filtrados)

def detectar_vales_focado(mask_pedras, gray_img):
    """
    Combinação focada nos métodos que funcionaram melhor
    """
    print("🎯 Detectando vales com métodos focados...")

    todos_pontos = []
    confiancas = []

    # Método 1: Curvatura Avançada (principal)
    # pontos_curvatura = detectar_vales_curvatura_avancado(mask_pedras, gray_img)
    # if len(pontos_curvatura) > 0:
    #     todos_pontos.extend(pontos_curvatura)
    #     # Dar peso maior para este método
    #     for _ in range(3):  # Triplicar a importância
    #         confiancas.extend([1.0] * len(pontos_curvatura))
    # print(f"  ✓ Curvatura Avançada: {len(pontos_curvatura)} vales")

    # Método 2: Multi-escala Otimizado (complementar)
    pontos_multiescala = detectar_vales_multi_escala_otimizado(mask_pedras, gray_img)
    if len(pontos_multiescala) > 0:
        todos_pontos.extend(pontos_multiescala)
        confiancas.extend([0.8] * len(pontos_multiescala))
    print(f"  ✓ Multi-escala Otimizado: {len(pontos_multiescala)} vales")

    # Método 3: Gradiente de Intensidade (bônus)
    # pontos_gradiente = detectar_vales_gradiente_intensidade(mask_pedras, gray_img)
    # if len(pontos_gradiente) > 0:
    #     todos_pontos.extend(pontos_gradiente)
    #     confiancas.extend([0.6] * len(pontos_gradiente))
    # print(f"  ✓ Gradiente de Intensidade: {len(pontos_gradiente)} vales")

    if not todos_pontos:
        return np.array([])

    # Agrupar pontos próximos considerando confiança
    pontos_final = agrupar_com_confianca(np.array(todos_pontos), np.array(confiancas), CONFIG_VALES['distancia_filtro'])
    print(f"  📊 Total após agrupamento inteligente: {len(pontos_final)} vales")

    return pontos_final

def agrupar_com_confianca(pontos, confiancas, raio=15):
    """
    Agrupa pontos considerando seus pesos de confiança
    """
    if len(pontos) == 0:
        return pontos

    grupos = []
    usados = set()

    for i in range(len(pontos)):
        if i in usados:
            continue

        grupo = [pontos[i]]
        pesos = [confiancas[i]]
        usados.add(i)

        # Expandir grupo
        for j in range(len(pontos)):
            if j not in usados:
                dist = np.linalg.norm(pontos[i] - pontos[j])
                if dist < raio:
                    grupo.append(pontos[j])
                    pesos.append(confiancas[j])
                    usados.add(j)

        # Calcular centroide ponderado pela confiança
        if sum(pesos) > 0:
            centroide = np.average(grupo, weights=pesos, axis=0).astype(np.int32)
            grupos.append(centroide)

    return np.array(grupos)

def visualizar_vales_detalhado(img, mask_pedras, pontos_vale, titulo="Vales Detectados"):
    """
    Visualização colorida por densidade de vales
    """
    img_debug = img.copy()

    # Mapa de calor dos vales
    heatmap = np.zeros(mask_pedras.shape, dtype=np.float32)

    for ponto in pontos_vale:
        cv2.circle(heatmap, tuple(ponto), CONFIG_VALES['distancia_filtro'], 1.0, -1)

    # Normalizar heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Aplicar colormap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    # Overlay na imagem
    mask_overlay = mask_pedras > 0
    img_debug[mask_overlay] = cv2.addWeighted(img_debug[mask_overlay], 0.3,
                                              heatmap_color[mask_overlay], 0.7, 0)

    # Desenhar contornos
    contours, _ = cv2.findContours(mask_pedras, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_debug, contours, -1, (0, 255, 0), 2)

    # Desenhar pontos de vale com numeração
    for i, ponto in enumerate(pontos_vale):
        # Círculo colorido baseado na posição
        cor = tuple(map(int, heatmap_color[ponto[1], ponto[0]]))
        cv2.circle(img_debug, tuple(ponto), 6, cor, -1)
        cv2.circle(img_debug, tuple(ponto), 8, (255, 255, 255), 2)

        # Número do vale
        cv2.putText(img_debug, str(i), (ponto[0]+10, ponto[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    cv2.imshow(titulo, img_debug)
    return img_debug

## Área de corte
def cortar_nos_vales_inteligente(mask_pedra_solida, pontos_vale, raio_proximidade=50, min_pontos_corte=2):
    """
    Corta a máscara nos vales detectados usando análise de proximidade e geometria

    Args:
        mask_pedra_solida: Máscara binária das pedras
        pontos_vale: Array de pontos (x,y) dos vales detectados
        raio_proximidade: Distância máxima para considerar dois vales como opostos
        min_pontos_corte: Mínimo de pontos para formar uma linha de corte

    Returns:
        Lista com contornos encontrados após cortes
    """
    if len(pontos_vale) < min_pontos_corte:
        print("⚠️ Pontos insuficientes para realizar cortes")
        return mask_pedra_solida

    mask_cortada = mask_pedra_solida.copy()
    h, w = mask_cortada.shape

    # 1. Encontrar o contorno principal para análise
    contours, _ = cv2.findContours(mask_pedra_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filtra pelo tamanho do contorno.
    # contours_filtrados = [cnt for cnt in contours if cv2.contourArea(cnt) >= CONFIG_VALES['area_min']]

    if not contours:
        return mask_cortada

    main_contour = max(contours, key=cv2.contourArea)

    # 2. Agrupar pontos por proximidade e posição relativa
    pares_corte = encontrar_pares_corte(pontos_vale, main_contour, raio_proximidade)

    # print(f"🔪 Encontrados {len(pares_corte)} pares de corte")

    # 3. Para cada par de vales, criar uma linha de corte
    linhas_corte = []
    for i, (p1, p2, score) in enumerate(pares_corte):
        # Calcular linha de corte estendida
        linha = calcular_linha_corte_otimizada(p1, p2, mask_pedra_solida, main_contour)
        if linha is not None:
            linhas_corte.append(linha)
            # print(f"  📏 Corte {i+1}: {p1} -> {p2} (score: {score:.2f})")

    # 4. Aplicar todos os cortes
    for linha in linhas_corte:
        # Desenhar linha de corte (valor 0 = preto)
        cv2.line(mask_cortada, linha[0], linha[1], 0, thickness=3)

        # Adicionar um pequeno círculo nos extremos para garantir o corte
        cv2.circle(mask_cortada, linha[0], 2, 0, -1)
        cv2.circle(mask_cortada, linha[1], 2, 0, -1)

    # 5. Verificar se os cortes foram efetivos
    contours_apos, _ = cv2.findContours(mask_cortada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"📦 Antes: {len(contours)} contorno(s) | Depois: {len(contours_apos)} contorno(s)")

    # 6. Se não separou, tentar cortes mais agressivos
    if len(contours_apos) == len(contours) and len(pontos_vale) > 0:
        print("⚠️ Cortes não foram suficientes, aplicando cortes agressivos...")
        mask_cortada = aplicar_cortes_agressivos(mask_cortada, pontos_vale, raio_proximidade)

    # return mask_cortada, contours_apos
    return contours_apos

def encontrar_pares_corte(pontos_vale, contorno, raio_max=50):
    """
    Encontra pares de vales que devem ser conectados para corte
    """
    pares = []
    pontos_usados = set()

    # Calcular centro do contorno
    M = cv2.moments(contorno)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0

    # Para cada ponto, encontrar o melhor par
    for i, p1 in enumerate(pontos_vale):
        if i in pontos_usados:
            continue

        melhor_par = None
        melhor_score = 0

        for j, p2 in enumerate(pontos_vale):
            if i == j or j in pontos_usados:
                continue

            # Calcular distância entre os pontos
            dist = np.linalg.norm(p1 - p2)

            if dist < raio_max:
                # Calcular score baseado em múltiplos fatores
                score = calcular_score_par(p1, p2, contorno, (cx, cy))

                # Verificar se os pontos estão em lados opostos
                vetor_centro_p1 = p1 - np.array([cx, cy])
                vetor_centro_p2 = p2 - np.array([cx, cy])

                # Produto escalar negativo indica lados opostos
                if np.dot(vetor_centro_p1, vetor_centro_p2) < 0:
                    score *= 1.5  # Bônus para lados opostos

                # Verificar se a linha entre eles cruza o contorno adequadamente
                if linha_cruza_contorno_corretamente(p1, p2, contorno):
                    score *= 1.3

                if score > melhor_score:
                    melhor_score = score
                    melhor_par = (p1, p2, score)

        if melhor_par is not None:
            pares.append(melhor_par)
            pontos_usados.add(i)
            # Encontrar o índice do par escolhido
            for j, p in enumerate(pontos_vale):
                if np.array_equal(p, melhor_par[1]):
                    pontos_usados.add(j)
                    break

    return pares

def calcular_score_par(p1, p2, contorno, centro):
    """
    Calcula um score de qualidade para um par de pontos de corte
    """
    score = 1.0

    # 1. Distância (preferir pares mais próximos)
    dist = np.linalg.norm(p1 - p2)
    score *= (1.0 / (1.0 + dist/100))  # Normalizar distância

    # 2. Verificar se a linha passa pelo interior do contorno
    linha = np.array([p1, p2])
    pontos_linha = gerar_pontos_linha(p1, p2)

    pontos_dentro = 0
    for ponto in pontos_linha:
        if cv2.pointPolygonTest(contorno, tuple(ponto.astype(float)), False) >= 0:
            pontos_dentro += 1

    proporcao_dentro = pontos_dentro / len(pontos_linha)
    score *= (0.5 + proporcao_dentro)

    # 3. Perpendicularidade com o eixo principal do contorno
    try:
        (x, y), (MA, ma), angle = cv2.fitEllipse(contorno)
        eixo_principal = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        vetor_corte = (p2 - p1) / np.linalg.norm(p2 - p1)

        # Queremos cortes perpendiculares ao eixo principal
        perpendicularidade = abs(np.dot(eixo_principal, vetor_corte))
        score *= (1.0 + perpendicularidade)
    except:
        pass

    return score

def calcular_linha_corte_otimizada(p1, p2, mask, contorno):
    """
    Calcula uma linha de corte que realmente separa o contorno
    """
    # Vetor direção
    direcao = p2 - p1
    comprimento = np.linalg.norm(direcao)

    if comprimento == 0:
        return None

    direcao_norm = direcao / comprimento

    # Perpendicular para estender a linha
    perp = np.array([-direcao_norm[1], direcao_norm[0]])

    # Estender a linha para garantir que corte completamente
    extensao = 20  # pixels

    p1_ext = (p1 - direcao_norm * extensao).astype(np.int32)
    p2_ext = (p2 + direcao_norm * extensao).astype(np.int32)

    # Ajustar para não sair da imagem
    h, w = mask.shape
    p1_ext[0] = np.clip(p1_ext[0], 0, w-1)
    p1_ext[1] = np.clip(p1_ext[1], 0, h-1)
    p2_ext[0] = np.clip(p2_ext[0], 0, w-1)
    p2_ext[1] = np.clip(p2_ext[1], 0, h-1)

    return (tuple(p1_ext), tuple(p2_ext))

def linha_cruza_contorno_corretamente(p1, p2, contorno):
    """
    Verifica se a linha entre p1 e p2 cruza o contorno de forma adequada
    """
    # Criar uma máscara temporária com a linha
    mask_temp = np.zeros((1000, 1000), dtype=np.uint8)  # Ajustar tamanho conforme necessário
    cv2.line(mask_temp, tuple(p1), tuple(p2), 255, 2)

    # Verificar interseção com o contorno
    intersecoes = 0
    pontos_linha = gerar_pontos_linha(p1, p2)

    for ponto in pontos_linha:
        dist = cv2.pointPolygonTest(contorno, tuple(ponto.astype(float)), True)
        if abs(dist) < 2:  # Próximo à borda
            intersecoes += 1

    # Esperamos pelo menos 2 interseções (entrada e saída)
    return intersecoes >= 2

def gerar_pontos_linha(p1, p2, num_pontos=20):
    """
    Gera pontos ao longo de uma linha entre p1 e p2
    """
    pontos = []
    for t in np.linspace(0, 1, num_pontos):
        ponto = p1 * (1-t) + p2 * t
        pontos.append(ponto.astype(np.int32))
    return pontos

def aplicar_cortes_agressivos(mask, pontos_vale, raio=50):
    """
    Aplica cortes mais agressivos quando os normais não funcionam
    """
    mask_cortada = mask.copy()
    h, w = mask.shape

    # Agrupar pontos por proximidade espacial
    grupos = agrupar_pontos_espaciais(pontos_vale, raio)

    # Para cada grupo, criar um corte em estrela
    for grupo in grupos:
        if len(grupo) >= 2:
            # Calcular centro do grupo
            centro = np.mean(grupo, axis=0).astype(np.int32)

            # Criar cortes radiais a partir do centro
            for ponto in grupo:
                # Estender a linha do centro até o ponto
                direcao = ponto - centro
                dist = np.linalg.norm(direcao)

                if dist > 0:
                    direcao_norm = direcao / dist
                    ponto_ext = (centro + direcao_norm * (dist + 30)).astype(np.int32)

                    # Desenhar linha de corte mais grossa
                    cv2.line(mask_cortada, tuple(centro), tuple(ponto_ext), 0, thickness=5)

    return mask_cortada

def agrupar_pontos_espaciais(pontos, raio):
    """
    Agrupa pontos que estão próximos espacialmente
    """
    if len(pontos) == 0:
        return []

    grupos = []
    pontos_restantes = list(range(len(pontos)))

    while pontos_restantes:
        idx_atual = pontos_restantes.pop(0)
        grupo_atual = [pontos[idx_atual]]

        # Encontrar todos os pontos próximos
        i = 0
        while i < len(pontos_restantes):
            idx = pontos_restantes[i]
            dist = np.linalg.norm(grupo_atual[0] - pontos[idx])

            if dist < raio:
                grupo_atual.append(pontos[idx])
                pontos_restantes.pop(i)
            else:
                i += 1

        grupos.append(np.array(grupo_atual))

    return grupos

# Função de visualização dos cortes
def visualizar_cortes(img_original, mask_original, mask_cortada, contornos, pares_corte, titulo="Análise de Cortes"):
    """
    Visualiza os cortes aplicados na máscara
    """
    # Criar imagem de debug
    debug_img = img_original.copy()

    # Overlay da máscara original em azul
    overlay_original = np.zeros_like(debug_img)
    overlay_original[mask_original > 0] = [255, 0, 0]  # Azul
    debug_img = cv2.addWeighted(debug_img, 0.7, overlay_original, 0.2, 0)

    # Overlay da máscara cortada em verde
    overlay_cortada = np.zeros_like(debug_img)
    overlay_cortada[mask_cortada > 0] = [0, 255, 0]  # Verde
    debug_img = cv2.addWeighted(debug_img, 1.0, overlay_cortada, 0.2, 0)

    # Desenhar linhas de corte
    for p1, p2, score in pares_corte:
        # Linha de corte em vermelho
        cv2.line(debug_img, tuple(p1), tuple(p2), (0, 0, 255), 2)

        # Pontos dos vales
        cv2.circle(debug_img, tuple(p1), 5, (255, 255, 0), -1)
        cv2.circle(debug_img, tuple(p2), 5, (255, 255, 0), -1)

        # Score do par
        centro = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
        cv2.putText(debug_img, f"{score:.1f}", centro,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)

    # Contar objetos antes e depois
    contours_antes, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if 4500 > area > 800:
            cv2.drawContours(debug_img, [cnt], -1, 255, thickness=-1)

    # Adicionar texto informativo
    cv2.putText(debug_img, f"Antes: {len(contours_antes)} objetos", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Depois: {len(contornos)} objetos", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Cortes: {len(pares_corte)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(titulo, debug_img)
    return debug_img



# O resto do seu código continua igual...
# TESTE
# pipeline_blackhat("imagem_recortada.jpeg")
# pipeline_blackhat("imagem.jpeg")
pipeline_blackhat(args.imagem)
