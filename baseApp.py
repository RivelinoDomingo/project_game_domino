import cv2
import numpy as np
import math
import argparse
import time
# from skimage.morphology import skeletonize
import sys
# from scipy.signal import find_peaks


def parse_arguments():
    parser = argparse.ArgumentParser(description='Processa imagens de dominó')
    parser.add_argument('imagem', help='Caminho para o arquivo de imagem')
    parser.add_argument('-z', '--zoom', type=float, default=1.0, help='Nível de zoom (padrão 1.0)')
    parser.add_argument('-p', '--proximidade', type=int, default=37, help='Distância mínima entre pedras')
    parser.add_argument('-d', '--debug', action='store_true', help='Ativa modo depuração')
    return parser.parse_args()


CONFIGS = {
    'distancia_filtro': 20,
    'distancia_corte': 62,
    'distancia_conexao': 600,
    'tamanho_kernel_morfologia': 15, # Novo parâmetro para o tamanho da fenda a ser fechada
    'area_max': 1600,                # Area maxima das pedras
    'area_min': 500,
    'area_ponto': 35,
}


def pipeline_blackhat(args):
    time_start = time.time()
    img = cv2.imread(args.imagem)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return

    # Área de zoom
    img = cv2.resize(img, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

   # 1. Máscara Sólida Base
    _, mask_branca = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    contours_ext, _ = cv2.findContours(mask_branca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow("1 - Mask Branca", mask_branca)

    mask_solida = np.zeros_like(gray)
    cv2.drawContours(mask_solida, contours_ext, -1, 255, thickness=cv2.FILLED)

    # mask_branca = cv2.medianBlur(mask_branca, 3)
    # cv2.imshow("1 - Mask Branca 120", mask_branca)
    # mask_solida = cv2.medianBlur(mask_solida, 3)
    # cv2.imshow("1 - Mask Branca 190", mask_branca_1)


    # Refinamento de Contornos
    cnts_pre, _ = cv2.findContours(mask_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtrada = np.zeros_like(gray)

    # Melhoria
    # fator_area = args.zoom ** 2
    fator_area = 1.4 ** 2
    area_min = int(CONFIGS['area_min'] * fator_area)
    area_max = int(CONFIGS['area_max'] * fator_area)
    raio_corte = int(CONFIGS['distancia_corte'] * args.zoom) # Distância é linear

    for c in cnts_pre:
        if cv2.contourArea(c) > area_min:
            cv2.drawContours(mask_filtrada, [c], -1, 255, -1)


    # Para remover os pontos pretos (subtrair áreas)
    mask_diferenca = cv2.bitwise_and(mask_filtrada, mask_branca)

    # Se preferir ver onde os pontos foram removidos:
    mask_pontos = cv2.bitwise_xor(mask_filtrada, mask_branca)

    # cv2.imshow("0 -- Mask Cortada", mask_diferenca)
    # cv2.imshow("0 -- Mask Removidos", mask_pontos)
    # cv2.waitKey()
    # sys.exit(0)

    pontos_vale = detectar_vales_por_morfologia(mask_filtrada)

    kernel_derreter = np.ones((7, 7), np.uint8)
    mask_corte = cv2.erode(mask_filtrada, kernel_derreter, iterations=3)

    # cv2.imshow("0 -- Mask Filtrada", mask_filtrada)


    contours_final, _ = cv2.findContours(mask_filtrada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt_finais = []
    if len(pontos_vale) >= 2 and contours_final:
        if args.debug:
            img_debug_final = visualizar_vales_detalhado(img, mask_filtrada, pontos_vale)

        # if args.debug:
        #     cv2.imshow("2 - Mask Usada", mask_filtrada)

        pares_corte = encontrar_pares_corte(pontos_vale, mask_filtrada, raio_corte)
        cnt_finais = cortar_nos_vales_inteligente(mask_filtrada, pares_corte)
        max_contorno = max(cnt_finais, key=cv2.contourArea)
        max_contorno = cv2.contourArea(max_contorno)
        # print(f"Maior contorno: {max_contorno}  -- Menor: {min_contorno}")

        if max_contorno > area_max * 3:
            print("Recalculando com mascara reduzinda!")
            print(f"Área max. permitida: {area_max * 1.8}")
            print(f"Área do maior Contorno: {max_contorno}")
            # contours_corte, _ = cv2.findContours(mask_corte, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # mask_corte = np.zeros_like(gray)
            # contorno_corte = max(contours_corte, key=cv2.contourArea)
            # cv2.drawContours(mask_corte, [contorno_corte], -1, 255, -1)

            pares_corte = encontrar_pares_corte(pontos_vale, mask_corte, raio_corte)
            cnt_finais = cortar_nos_vales_inteligente(mask_filtrada, pares_corte)
            if args.debug:
                cv2.imshow("2 - Mask Corte Usada", mask_corte)

        if args.debug:
            mask_cortada = np.zeros_like(gray)
            cv2.drawContours(mask_cortada, cnt_finais, -1, 255, thickness=cv2.FILLED)
            cv2.imshow("2 - Mask Cortada", mask_cortada)
            visualizar_cortes(img_debug_final, mask_filtrada, cnt_finais, pares_corte, "2 - Cortes Aplicados")

    else:
        if contours_final:
            cnt_finais = contours_final
        print("⚠️ Poucos vales detectados ou nenhum contorno encontrado!")


    candidatos = []
    out = img.copy()

    if args.debug:
        print(f"Contornos encontrados: {len(cnt_finais)}")

    for cnt in cnt_finais:
        area = cv2.contourArea(cnt)
        # if area < 10 or area > 2200:
        #     continue
        if area_max > area > area_min * 0.7:
            rect = cv2.minAreaRect(cnt)
            center, size, angle = rect
            w_box, h_box = size

            if w_box == 0 or h_box == 0:
                # print("Box com dimensões zeradas")
                continue

            width, height = size

            if width > height:
                ratio = width/height
                margem_A = 0.9
                margem_L = 1.02

            else:
                ratio = height/width
                margem_A = 1.05
                margem_L = 0.95

            # print(f"Valor de ratio: {ratio} e Área: {area}")

            if not (2.4 >= ratio >= 1.6):
                # print(f"Ratio fora do padrão: {ratio}")
                continue

            rect_pedra = (center, (width*margem_L, height*margem_A), angle)

            candidatos.append({
                'rect_pedra': rect_pedra,
                'centro': center
            })
        else:
            if args.debug:
                print(f"Área fora do range: {area}")
            pass

    if args.debug:
        print(f"Candidatos aprovados: {len(candidatos)}")



    pedras_unicas = []

    for cand in candidatos:
        cx1, cy1 = cand['centro']
        duplicata = False

        for p in pedras_unicas:
            cx2, cy2 = p['centro']
            if math.hypot(cx2 - cx1, cy2 - cy1) < args.proximidade:
                duplicata = True
                break

        if not duplicata:
            pedras_unicas.append(cand)


    # ==========================================
    # PASSO 2: O Filtro da "Área de Influência" (O Maior Bando)
    # ==========================================
    DISTANCIA_CONEXAO = CONFIGS['distancia_conexao'] * args.zoom  # Tamanho da "Área de influência" de cada pedra

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

    for d in pedras_aprovadas[:]:
        # Enviamos a imagem original limpa (img) e o retângulo da pedra
        pts_cima, pts_baixo, zero = extrair_e_contar(mask_pontos, d['rect_pedra'])
        texto = f"{pts_cima}|{pts_baixo}"
        if texto == "0|0" and not zero:
            pedras_aprovadas.remove(d)
            continue
        box_pedra = np.int32(cv2.boxPoints(d['rect_pedra']))
        cv2.drawContours(out, [box_pedra], 0, (0, 255, 0), 2)
        # Escrever o resultado na imagem, do lado da pedra!

        cx, cy = int(d['centro'][0]), int(d['centro'][1])

        # Fundo preto para o texto ficar legível
        cv2.putText(out, texto, (cx - 20, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        # Texto em branco
        cv2.putText(out, texto, (cx - 20, cy - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        print(f"Pedra encontrada: {texto}")

    print(f"Pedras aprovadas: {len(pedras_aprovadas)}")
    time_end = time.time() - time_start
    print(f"Tempo de duração da execução: {time_end}")
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
    box = cv2.boxPoints(rect_pedra)
    pts = ordenar_pontos(box)

    dist_0_1 = np.linalg.norm(pts[0] - pts[1])
    dist_0_3 = np.linalg.norm(pts[0] - pts[3])

    if dist_0_1 > dist_0_3:
        pts = np.array([pts[1], pts[2], pts[3], pts[0]], dtype="float32")

    dst = np.array([[0, 0], [39, 0], [39, 79], [0, 79]], dtype="float32")
    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(img, M, (40, 80))

    warped = cv2.medianBlur(warped, 3)
    _, thresh = cv2.threshold(warped, 160, 255, cv2.THRESH_BINARY)
    # kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
    # warped = cv2.morphologyEx(warped, cv2.MORPH_CLOSE, kernel_fechamento)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    zero_local = False
    if contornos:
        area = 0.0
        for c in contornos:
            area += cv2.contourArea(c)
            if area >= CONFIGS['area_ponto']:
                zero_local = True
                break


    metade_cima = thresh[0:40, 0:40]
    metade_baixo = thresh[40:80, 0:40]

    # cv2.imshow("0 -- Meio pedra", warped)
    # cv2.waitKey()

    def contar_bolinhas(metade):
        # gray = cv2.cvtColor(metade, cv2.COLOR_BGR2GRAY)
        # thresh = cv2.adaptiveThreshold(metade, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        # h, w = thresh.shape
        # cv2.rectangle(thresh, (0, 0), (w, h), 0, 3)

        # kernel = np.ones((2, 2), np.uint8)
        # kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_fechamento)
        # thresh = cv2.medianBlur(thresh, 7)

        # cv2.imshow("0 -- Meio pedra", thresh)
        # cv2.waitKey()

        contornos, _ = cv2.findContours(metade, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pontos = 0
        # CORREÇÃO: Usando o zoom ao quadrado
        point_area = CONFIGS['area_ponto']

        for c in contornos:
            area = cv2.contourArea(c)
            circularidade = 0.0
            if point_area * 0.6 < area < point_area * 2.1:
                perimetro = cv2.arcLength(c, True)
                if perimetro == 0:
                    continue
                circularidade = 4 * np.pi * (area / (perimetro * perimetro))
                if circularidade >= 0.7:
                    pontos += 1
            # print(f"Valor de Área={area}, Circularidade={circularidade}")

        return min(pontos, 6)

    pts_cima  = contar_bolinhas(metade_cima)
    pts_baixo = contar_bolinhas(metade_baixo)

    return pts_cima, pts_baixo, zero_local

# ====================================================================
# SUBSISTEMA DE LOCALIZAÇÂO DE VALES
# ====================================================================

def detectar_vales_por_morfologia(mask_solida):
    """
    Aplica a ideia de preencher as fendas e subtrair a imagem original
    para isolar os vales.
    """
    # 1. 'Massa Corrida' (Fechamento)
    k_size = CONFIGS['tamanho_kernel_morfologia']
    kernel_fechamento = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    mask_fechada = cv2.morphologyEx(mask_solida, cv2.MORPH_CLOSE, kernel_fechamento)

    # 2. Subtração (O Pulo do Gato)
    mask_vales = cv2.subtract(mask_fechada, mask_solida)
    if args.debug:
        cv2.imshow("1 - Mask Solida", mask_solida) # Descomente se precisar debugar
        cv2.imshow("1 - Mask Vales", mask_vales) # Descomente se precisar debugar

    # 3. Extrair os Pontos
    cnts_vales, _ = cv2.findContours(mask_vales, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pontos_encontrados = []
    for c in cnts_vales:
        cx, cy = cv2.minAreaRect(c)[0]
        area = cv2.contourArea(c)
        # Correção: passar como uma tupla contendo a coordenada e a área
        pontos_encontrados.append(([cx, cy], area))

    return agrupar_pontos_proximos(pontos_encontrados, int(CONFIGS['distancia_filtro'] * args.zoom))

def agrupar_pontos_proximos(dados_pontos, raio=5):
    """
    Recebe uma lista no formato [ ([cx, cy], area), ... ]
    Ordena por área para garantir a sobrevivência do ponto mais forte
    sem perder a performance do NumPy.
    """
    if len(dados_pontos) == 0:
        return np.array([])

    # 1. O Pulo do Gato: Ordenar do maior para o menor (pela área)
    # Assim, o primeiro ponto de qualquer aglomeração SEMPRE será o "mais forte"
    dados_ordenados = sorted(dados_pontos, key=lambda x: x[1], reverse=True)

    # 2. Separar apenas as coordenadas para a matemática vetorial do NumPy
    pontos = np.array([item[0] for item in dados_ordenados])

    finais = []
    visitados = np.zeros(len(pontos), dtype=bool)

    for i in range(len(pontos)):
        if visitados[i]:
            continue

        # Como ordenamos antes, este p_atual é garantidamente o de MAIOR ÁREA na vizinhança
        p_atual = pontos[i]
        finais.append(p_atual.astype(int))

        # Magia do NumPy: Calcula a distância deste ponto para TODOS os outros de uma vez
        distancias = np.linalg.norm(pontos - p_atual, axis=1)

        # Marca como 'visitado' (descarta) todos que estiverem dentro do raio de tolerância.
        # Os que estão sendo descartados têm área menor ou igual ao p_atual.
        visitados[distancias < raio] = True

    return np.array(finais)
# ====================================================================
# SUBSISTEMA DE CORTES REFATORADO E OTIMIZADO
# ====================================================================

def encontrar_pares_corte(pontos_vale, mask_pedras, raio_max=69):
    """
    Vetorizado. Usa a máscara sólida em vez de polígonos para evitar
    o problema de pedras isoladas (ilhas) sendo ignoradas.
    """
    if len(pontos_vale) < 2:
        return []

    pontos = np.array(pontos_vale)
    n_pontos = len(pontos)

    diffs = pontos[:, np.newaxis, :] - pontos[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diffs, axis=-1)

    pares_candidatos = []
    altura_img, largura_img = mask_pedras.shape[:2]

    for i in range(n_pontos):
        for j in range(i + 1, n_pontos):
            dist = dist_matrix[i, j]

            if dist < raio_max:
                # 2. NOVA Verificação Rápida e Robusta de Interseção:
                # Testamos 3 pontos internos ao longo da linha (25%, 50% e 75%)
                # para evitar falsos negativos caso a pedra tenha bordas irregulares.

                pA = pontos[i]
                pB = pontos[j]

                # Fatores de interpolação (o quão longe estamos de A em direção a B)
                fracoes = [0.25, 0.50, 0.75]

                linha_valida = False
                for f in fracoes:
                    # Calcula o ponto exato naquela fração da linha
                    pt_amostra = pA + (pB - pA) * f
                    mx, my = int(pt_amostra[0]), int(pt_amostra[1])

                    # Checagem de segurança dos limites da imagem
                    if 0 <= my < altura_img and 0 <= mx < largura_img:
                        # Se achou pelo menos um pixel branco forte, a linha cruza a pedra
                        if mask_pedras[my, mx] > 0:
                            linha_valida = True
                            break # Otimização: não precisa testar as outras frações

                # Se após testar os 3 pontos todos caíram no fundo preto, descarta o par.
                if not linha_valida:
                    continue

                # Pontuação base (distância)
                score = 100.0 / (1.0 + dist)

                # 3. Triangulação (Identificar ângulos +- 90°)
                bonus_triangulacao = 1.0
                # dist_AB = dist

                for k in range(n_pontos):
                    if k != i and k != j:
                        dist_AC = dist_matrix[i, k]

                        # Usando a sua margem de proporção testada
                        dist_real = 65 * args.zoom
                        # if (dist_AB * 1.7) < dist_AC < (dist_AB * 3.2):
                        if dist_real > dist_AC > dist_real * 0.5:
                            # print(f"Corte Dentro do range: dist_AB: {dist_AB} -- dist_AC: {dist_AC}")
                            v1 = pontos[j] - pontos[i]
                            v2 = pontos[k] - pontos[i]
                            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)

                            if n1 > 0 and n2 > 0:
                                cos_theta = np.dot(v1, v2) / (n1 * n2)
                                if abs(cos_theta) < 0.35:
                                    bonus_triangulacao = 2.0
                                    break
                #         # else:
                        #     print(f"Corte fora do range: dist_AB: {dist_AB} -- dist_AC: {dist_AC}")
                score *= bonus_triangulacao
                pares_candidatos.append((i, j, score))

    # Ordenar pelos melhores cortes
    pares_candidatos.sort(key=lambda x: x[2], reverse=True)

    # Evitar reutilização de pontos
    pares_finais = []
    pontos_usados = set()

    for i, j, score in pares_candidatos:
        if i not in pontos_usados and j not in pontos_usados:
            pares_finais.append((pontos[i], pontos[j], score))
            pontos_usados.add(i)
            pontos_usados.add(j)

    return pares_finais

def cortar_nos_vales_inteligente(mask_pedra_solida, pares_corte):
    """
    Aplica as linhas de corte geradas pelo algoritmo otimizado.
    """
    if not pares_corte:
        contours, _ = cv2.findContours(mask_pedra_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    mask_cortada = mask_pedra_solida.copy()

    for p1, p2, _ in pares_corte:
        cv2.line(mask_cortada, p1, p2, 0, thickness=2)
        cv2.circle(mask_cortada, p1, 2, 0, -1)
        cv2.circle(mask_cortada, p2, 2, 0, -1)

    contours_apos, _ = cv2.findContours(mask_cortada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    return contours_apos

# ====================================================================
# FUNÇÕES DE DEBUG VISUAL (Mantidas inalteradas)
# ====================================================================

# Função de visualização dos cortes
def visualizar_cortes(img_original, mask_original, contornos, pares_corte, titulo="Análise de Cortes"):
    """
    Visualiza os cortes aplicados na máscara
    """
    # Criar imagem de debug
    debug_img = img_original.copy()

    # Desenhar linhas de corte
    for p1, p2, score in pares_corte:
        # Linha de corte em vermelho
        cv2.line(debug_img, tuple(p1), tuple(p2), (0, 0, 255), 2)

        # Pontos dos vales
        cv2.circle(debug_img, tuple(p1), 5, (0, 255, 0), -1)
        cv2.circle(debug_img, tuple(p2), 5, (0, 255, 0), -1)

        # Score do par
        centro = ((p1[0] + p2[0])//2, (p1[1] + p2[1])//2)
        cv2.putText(debug_img, f"{score:.1f}", centro,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

    # Contar objetos antes e depois
    contours_antes, _ = cv2.findContours(mask_original, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fator_area = args.zoom ** 2
    area_min = int(CONFIGS['area_min'] * fator_area)
    area_max = int(CONFIGS['area_max'] * fator_area)

    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area_max > area > area_min:
            cv2.drawContours(debug_img, [cnt], -1, 255, thickness=-1)
            center, _, _ = cv2.minAreaRect(cnt)
            # Converte o centro para inteiros
            cx = int(center[0])
            cy = int(center[1])

            # Agora usa a tupla de inteiros
            cv2.putText(debug_img,
                        f"{area:.0f}",           # arredonda a área
                        (cx, cy),                   # ← aqui está o fix
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.3,
                        (255, 255, 255),
                        1)

    # Adicionar texto informativo
    cv2.putText(debug_img, f"Antes: {len(contours_antes)} objetos", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Depois: {len(contornos)} objetos", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(debug_img, f"Cortes: {len(pares_corte)}", (10, 90),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(titulo, debug_img)
    return debug_img

def visualizar_vales_detalhado(img, mask_pedras, pontos_vale, titulo="Vales Detectados"):
    """
    Visualização colorida por densidade de vales
    """
    img_debug = img.copy()

    # Mapa de calor dos vales
    heatmap = np.zeros(mask_pedras.shape, dtype=np.float32)

    for ponto in pontos_vale:
        cv2.circle(heatmap, tuple(ponto), int(CONFIGS['distancia_filtro'] * args.zoom), 1.0, -1)

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


# pipeline_blackhat("imagem_recortada.jpeg")
# pipeline_blackhat("imagem.jpeg")
if __name__ == "__main__":
    args = parse_arguments()
    pipeline_blackhat(args)
