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
    parser.add_argument('-L', '--limiar', type=int, default=190, help='Limiar de branco (0-255), valores de uso 150-200')
    parser.add_argument('-d', '--debug', nargs='*', default=None, help='Ativa debug. Códigos: Pn=pontos, Mb=mascaras, Tr=tracos, Vl=vales')
    return parser.parse_args()


CONFIGS = {
    'distancia_filtro': 15,
    'distancia_corte': 62,
    'distancia_conexao': 600,
    'tamanho_kernel_morfologia': 15, # Novo parâmetro para o tamanho da fenda a ser fechada
    'area_max': 2000,                # Area maxima das pedras
    'area_min': 300,
    'area_ponto': 15,
}


def debug_ativo(categoria=None):
    """
    Sem categoria: verifica se qualquer debug está ativo.
    Com categoria: verifica se aquela categoria específica está ativa,
    ou se foi chamado -d sem argumentos (debug geral).
    Códigos: Pn=pontos, Mb=mascaras, Tr=tracos, Vl=vales'.
    """
    if args.debug is None:
        return False
    if categoria is None:
        return True
    # -d sem argumentos = debug geral (ativa tudo)
    if len(args.debug) == 0:
        return True
    return categoria in args.debug

def validar_rect_na_mask(rect_pedra, mask_filtrada, limiar_ocupacao=0.80):
    """
    Verifica se o rect_pedra cobre ao menos limiar_ocupacao (80%) de pixels
    brancos na mask_filtrada. Retorna True se for uma pedra válida.
    """
    # Cria máscara do rect rotacionado
    mask_rect = np.zeros(mask_filtrada.shape, dtype=np.uint8)
    box = np.int32(cv2.boxPoints(rect_pedra))
    cv2.fillPoly(mask_rect, [box], 255)

    # Pixels dentro do rect
    total_pixels = cv2.countNonZero(mask_rect)
    if total_pixels == 0:
        return False

    # Pixels brancos na mask_filtrada dentro do rect
    intersecao = cv2.bitwise_and(mask_filtrada, mask_rect)
    pixels_brancos = cv2.countNonZero(intersecao)

    ocupacao = pixels_brancos / total_pixels
    return ocupacao >= limiar_ocupacao

def calcular_limiar_adaptativo(gray, args):
    if args.limiar != 190:  # usuário passou -L manualmente
        return args.limiar, gray

    # Percentil 85 dos pixels — representa a região mais clara da imagem
    # (o plástico branco das pedras puxa esse valor para cima)
    p85 = np.percentile(gray, 60)

    # Otsu restrito: analisa só os pixels ACIMA do percentil 85
    # Isso foca o histograma na transição fundo claro → plástico branco
    pixels_claros = gray[gray > p85]

    if len(pixels_claros) == 0:
        return args.limiar, gray

    # Histograma só dos pixels claros
    hist = np.bincount(pixels_claros.astype(np.uint8), minlength=256).astype(np.float32)

    # Otsu manual nessa faixa restrita
    total = pixels_claros.size
    soma = np.dot(np.arange(256), hist)
    soma_b, peso_b, maximo, limiar = 0.0, 0.0, 0.0, int(p85)

    for t in range(int(p85), 256):
        peso_b += hist[t]
        if peso_b == 0:
            continue
        peso_f = total - peso_b
        if peso_f == 0:
            break
        soma_b += t * hist[t]
        media_b = soma_b / peso_b
        media_f = (soma - soma_b) / peso_f
        variancia = peso_b * peso_f * (media_b - media_f) ** 2
        if variancia > maximo:
            maximo = variancia
            limiar = t

    # Garante que fica numa faixa razoável para dominó (160-220)
    limiar = int(np.clip(limiar, 160, 220))
    print(f"Limiar automático (Otsu restrito p85): {limiar}")
    return limiar, gray

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
    # _, mask_branca = cv2.threshold(gray, args.limiar, 255, cv2.THRESH_BINARY)
    limiar, gray = calcular_limiar_adaptativo(gray, args)
    _, mask_branca = cv2.threshold(gray, limiar, 255, cv2.THRESH_BINARY)
    contours_ext, _ = cv2.findContours(mask_branca, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask_solida = np.zeros_like(gray)
    cv2.drawContours(mask_solida, contours_ext, -1, 255, thickness=cv2.FILLED)

    # Refinamento de Contornos
    cnts_pre, _ = cv2.findContours(mask_solida, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filtrada = np.zeros_like(gray)

    # Melhoria
    fator_area = (args.zoom + 0.4) ** 2   # O + 0.4 por conta do zoom padrão de app.py
    # fator_area = 1.4 ** 2
    area_min = int(CONFIGS['area_min'] * fator_area)
    area_max = int(CONFIGS['area_max'] * fator_area)
    raio_corte = int(CONFIGS['distancia_corte'] * args.zoom) # Distância é linear

    for c in cnts_pre:
        if cv2.contourArea(c) > area_min:
            cv2.drawContours(mask_filtrada, [c], -1, 255, -1)

    if debug_ativo('Mb'):
        cv2.imshow("1 - Mask Branca", mask_branca)
        cv2.imshow("1 - Mask Filtrada", mask_filtrada)
        # cv2.imshow("1 - Mask Cinza", gray)

    # Alinhar contorno
    def alinhar_contorno(contorno):
        """
        Rotaciona o contorno pelo ângulo do minAreaRect
        para que o lado longo fique alinhado com o eixo Y (vertical).
        Retorna o boundingRect ajustado.
        """
        rect = cv2.minAreaRect(contorno)
        center, size, angle = rect
        w, h = size

        # Garante que h é sempre o lado longo
        if w > h:
            w, h = h, w
            angle += 90

        # Rotaciona os pontos do contorno em torno do centro
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Aplica a rotação nos pontos do contorno
        pontos = contorno.reshape(-1, 2).astype(np.float32)
        pontos_rot = cv2.transform(pontos.reshape(1, -1, 2), M).reshape(-1, 2)

        # boundingRect agora é ajustado ao eixo
        x = int(pontos_rot[:, 0].min())
        y = int(pontos_rot[:, 1].min())
        cw = int(pontos_rot[:, 0].max()) - x
        ch = int(pontos_rot[:, 1].max()) - y

        return x, y, cw, ch, angle

    # Se preferir ver onde os pontos foram removidos:
    mask_base_pontos = cv2.bitwise_xor(mask_filtrada, mask_branca)
    cnts_pontos, _ = cv2.findContours(mask_base_pontos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_esp_pontos = np.zeros_like(gray)
    mask_esp_tracos = np.zeros_like(gray)

    # Obter as médias
    med_cw_tracos = 0.0
    counter = 0

    for c in cnts_pontos:
        x, y, cw, ch, angle = alinhar_contorno(c)
        ratio = ch / cw if cw > 0 else 0
        if ratio > 2.5:
            med_cw_tracos += cw
            counter += 1

    med_cw_tracos = med_cw_tracos / counter if med_cw_tracos > 0 else 0
    print(f"Média dos Traços: {med_cw_tracos}")

    for c in cnts_pontos:
        x, y, cw, ch, angle = alinhar_contorno(c)

        ratio = ch / cw if cw > 0 else 0

        if ratio > 2.5 and cw < med_cw_tracos + 2:
            if angle > 100.0 or angle < 45.0:
                acr_x = x + 5
                acr_y = y + 10
            else:
                acr_x = x - 5
                acr_y = y + 2
            # cv2.putText(mask_esp_tracos, f"{int(angle)}", (acr_x, acr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, 255, 1)
            cv2.drawContours(mask_esp_tracos, [c], -1, 255, -1)
            # print(f"Traço ---- ratio={ratio:.2f} cw={cw} ch={ch} angle={angle:.1f}")
            # cv2.imshow("Contornos", mask_esp_tracos)
            # cv2.waitKey(0)
        else:
            cv2.drawContours(mask_esp_pontos, [c], -1, 255, -1)
            # print(f"     Ponto ----  ratio={ratio:.2f} cw={cw} ch={ch} angle={angle:.1f}")

        cnts_tracos, _ = cv2.findContours(
            mask_esp_tracos,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

    # sys.exit(0)

    def angle_diff(a, b):
        d = abs(a - b) % 180
        return min(d, 180 - d)

    fragmentos = []

    for c in cnts_tracos:

        x, y, cw, ch, angle = alinhar_contorno(c)

        cx = x + cw // 2
        cy = y + ch // 2

        fragmentos.append({
            'contour': c,
            'x': x,
            'y': y,
            'w': cw,
            'h': ch,
            'cx': cx,
            'cy': cy,
            'angle': angle
        })

    mask_tracos_unidos = np.zeros_like(gray)

    for frag in fragmentos:
        cv2.drawContours(
            mask_tracos_unidos,
            [frag['contour']],
            -1,
            255,
            -1
        )
    for i in range(len(fragmentos)):

        for j in range(i + 1, len(fragmentos)):

            a = fragmentos[i]
            b = fragmentos[j]

            # diferença vertical pequena
            dy = abs(a['cy'] - b['cy'])

            # distância horizontal
            dx = abs(a['cx'] - b['cx'])

            # ângulo parecido
            da = angle_diff(
                a['angle'],
                b['angle']
            )

            right_a = a['x'] + a['w']
            left_b  = b['x']

            gap = left_b - right_a

            if (
                dy < 20
                and dx < 20
                and da < 10
            ):

                pt1 = (a['cx'], a['cy'])
                pt2 = (b['cx'], b['cy'])

                cv2.line(
                    mask_tracos_unidos,
                    pt1,
                    pt2,
                    255,
                    thickness=1
                )
        # print(f"ratio={ratio:.2f} cw={cw} ch={ch} angle={angle:.1f}")
        # cv2.imshow("0 -- Mask Tracos", mask_esp_tracos)
        # cv2.waitKey()
    # ====================================================================
    # DETECÇÃO DE PEDRAS POR TRAÇOS (MODO EXPERIMENTAL)
    # Usa mask_esp_tracos para localizar o traço central de cada pedra
    # e constrói rect_pedra diretamente — sem depender de vales ou proximidade.
    # O pipeline de vales original continua intacto abaixo.
    # ====================================================================
    kernel_erode = np.ones((4, 4), np.uint8)
    kernel_close = np.ones((4, 4), np.uint8)
    mask_pontos_erode = cv2.erode(mask_esp_pontos, kernel_erode, iterations=1)

    # ============================================================
    # DISTANCE TRANSFORM
    # ============================================================

    # Garantir imagem binária
    mask_bin = (mask_esp_pontos > 0).astype(np.uint8)

    # Distance transform
    dist = cv2.distanceTransform(
        mask_bin,
        cv2.DIST_L2,
        3
    )

    # Visualização debug
    if debug_ativo('Tr'):
        dist_show = cv2.normalize(
            dist,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(np.uint8)

        cv2.imshow("DIST TRANSFORM", dist_show)

    # Threshold dos picos
    _, mask_pontos_sep = cv2.threshold(
        dist,
        0.38 * dist.max(),
        255,
        cv2.THRESH_BINARY
    )

    mask_pontos_sep = np.uint8(mask_pontos_sep)

    # Normaliza dist para 0-255 uint8
    # dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    # otsu_val, mask_pontos_sep = cv2.threshold(dist_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f"OtsuVal: {otsu_val:.0f}")

    # mask_pontos_sep = np.uint8(mask_pontos_sep)

    # Pequena dilatação para recuperar formato
    # kernel_restore = cv2.getStructuringElement(
    #     cv2.MORPH_ELLIPSE,
    #     (3, 3)
    # )
    #
    # mask_pontos_sep = cv2.dilate(
    #     mask_pontos_sep,
    #     kernel_restore,
    #     iterations=1
    # )

    # Fecha fragmentos do traço interrompidos pelo pino de aço
    # Kernel horizontal: une fragmentos ao longo do comprimento do traço
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    mask_tracos_close = cv2.morphologyEx(mask_esp_tracos, cv2.MORPH_CLOSE, kernel_h, iterations=3)
    mask_tracos_close = cv2.morphologyEx(mask_tracos_close, cv2.MORPH_CLOSE, kernel_close, iterations=1)
    mask_pontos = cv2.bitwise_or(mask_tracos_unidos, mask_pontos_sep)


    if debug_ativo('Tr'):
        cv2.imshow("TRACO -- Mask Pontos separados", mask_pontos_sep)
        # cv2.imshow("TRACO -- Mask Tracos fechados", mask_tracos_close)
        cv2.imshow("TRACO -- Tracos fechados Aling", mask_tracos_unidos)
        cv2.imshow("TRACO -- Mask Tracos e Pontos", mask_pontos)

    # --- Coleta contornos dos traços e calcula média do comprimento ---
    cnts_tracos, _ = cv2.findContours(mask_tracos_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    comprimentos = []
    for c in cnts_tracos:
        r = cv2.minAreaRect(c)
        _, (w_t, h_t), _ = r
        comp = max(w_t, h_t)
        esp  = min(w_t, h_t)
        if esp == 0:
            continue
        ratio_t = comp / esp
        # Traço válido: comprido e fino
        if ratio_t > 3.0 and comp > 8 * args.zoom:
            comprimentos.append(comp)

    if comprimentos:
        med_comp_traco = float(np.median(comprimentos))
        print(f"TRACO -- Mediana comprimento dos traços: {med_comp_traco:.1f}px")

        # Pedra de dominó: traço central ≈ 85% da largura interna da pedra
        # Ratio pedra ≈ 2:1  →  altura_pedra ≈ 2 * largura_pedra
        # largura_pedra ≈ comp_traco / 0.85
        # altura_pedra  ≈ largura_pedra * 2
        largura_pedra_est = med_comp_traco / 0.80
        altura_pedra_est  = largura_pedra_est * 2.0

        # Margem extra para não cortar as bolinhas nas bordas
        margem_extra = 1.15
        largura_final = largura_pedra_est * margem_extra
        altura_final  = altura_pedra_est  * margem_extra

        candidatos_traco = []

        for c in cnts_tracos:
            r = cv2.minAreaRect(c)
            (cx_t, cy_t), (w_t, h_t), angle_t = r
            comp = max(w_t, h_t)
            esp  = min(w_t, h_t)
            if esp == 0:
                continue
            ratio_t = comp / esp

            if ratio_t < 3.0 or not (med_comp_traco * 0.6 <= comp <= med_comp_traco * 1.4):
                continue

            # Usa alinhar_contorno para obter o ângulo real do lado longo
            # independente de como o minAreaRect ordenou w/h
            _, _, cw_al, ch_al, angle_alinhado = alinhar_contorno(c)
            # angle_alinhado já aponta para o lado longo (ch > cw após alinhamento)
            # rect_pedra: largura_final é o lado curto, altura_final é o lado longo
            # o ângulo do minAreaRect para o lado longo = angle_alinhado - 90
            angle_pedra = angle_alinhado - 90

            rect_pedra_traco = ((cx_t, cy_t), (largura_final, altura_final), angle_pedra)

            if not validar_rect_na_mask(rect_pedra_traco, mask_filtrada):
                continue  # rect mal orientado ou fora da pedra real

            candidatos_traco.append({
                'rect_pedra': rect_pedra_traco,
                'centro': (cx_t, cy_t)
            })

            # if debug_ativo('Tr):
            #     box_t = np.int32(cv2.boxPoints(r))
            #     box_p = np.int32(cv2.boxPoints(rect_pedra_traco))
            #     img_tr = img.copy()
            #     cv2.drawContours(img_tr, [box_t], 0, (255, 0, 0), 1)   # traço azul
            #     cv2.drawContours(img_tr, [box_p], 0, (0, 255, 0), 2)   # pedra verde
            #     cv2.imshow("TRACO -- Debug por traço", img_tr)
            #     cv2.waitKey(0)

        # --- Deduplicação por proximidade (igual ao pipeline de vales) ---
        pedras_traco_unicas = []
        for cand in candidatos_traco:
            cx1, cy1 = cand['centro']
            dup = False
            for p in pedras_traco_unicas:
                cx2, cy2 = p['centro']
                if math.hypot(cx2 - cx1, cy2 - cy1) < args.proximidade:
                    dup = True
                    break
            if not dup:
                pedras_traco_unicas.append(cand)

        # --- Maior bando (igual ao pipeline de vales) ---
        DIST_CONEXAO_TRACO = CONFIGS['distancia_conexao'] * args.zoom
        visitados_t = set()
        bandos_t = []

        for i, p1 in enumerate(pedras_traco_unicas):
            if i in visitados_t:
                continue
            bando = [p1]
            visitados_t.add(i)
            fila = [p1]
            while fila:
                foco = fila.pop(0)
                cx_f, cy_f = foco['centro']
                for j, p2 in enumerate(pedras_traco_unicas):
                    if j not in visitados_t:
                        cx2, cy2 = p2['centro']
                        if math.hypot(cx2 - cx_f, cy2 - cy_f) <= DIST_CONEXAO_TRACO:
                            bando.append(p2)
                            visitados_t.add(j)
                            fila.append(p2)
            bandos_t.append(bando)

        pedras_traco_aprovadas = max(bandos_t, key=len) if bandos_t else []
        pedras_traco_aprovadas.sort(key=lambda x: x['centro'][1])

        print(f"TRACO -- Pedras detectadas por traço: {len(pedras_traco_aprovadas)}")

        # --- Resultado visual lado a lado com o pipeline de vales ---
        out_traco = img.copy()
        med_area_traco = 0.0

        for d in pedras_traco_aprovadas[:]:
            pts_cima, pts_baixo, zero, med_ar = extrair_e_contar(mask_pontos, d['rect_pedra'])
            texto = f"{pts_cima}|{pts_baixo}"
            if texto == "0|0" and not zero:
                pedras_traco_aprovadas.remove(d)
                continue
            med_area_traco += med_ar
            box_p = np.int32(cv2.boxPoints(d['rect_pedra']))
            cv2.drawContours(out_traco, [box_p], 0, (0, 255, 0), 2)
            cx, cy = int(d['centro'][0]), int(d['centro'][1])
            cv2.putText(out_traco, texto, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(out_traco, texto, (cx - 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            print(f"TRACO -- Pedra: {texto}")

        if med_area_traco > 0 and pedras_traco_aprovadas:
            med_area_traco = abs(med_area_traco / len(pedras_traco_aprovadas))
        cv2.putText(out_traco, f"Area media Pontos: {med_area_traco:.2f}", (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("TRACO -- Resultado por Tracos", out_traco)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        print("TRACO -- Nenhum traço válido encontrado, usando apenas pipeline de vales.")


def extrair_e_contar(img, rect_pedra):
    center, size, angle = rect_pedra
    cx, cy = center

    w, h = size
    if w > h:
        w, h = h, w
        angle += 90

    w_int = int(round(w))
    h_int = int(round(h))

    # --- Rotaciona com INTER_NEAREST para preservar binário sem blur ---
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    altura_img, largura_img = img.shape[:2]
    img_rot = cv2.warpAffine(img, M, (largura_img, altura_img),
                             flags=cv2.INTER_NEAREST)  # <-- SEM interpolação

    # Recorta exatamente o bounding rect
    x1 = max(0, int(round(cx - w_int / 2)))
    y1 = max(0, int(round(cy - h_int / 2)))
    x2 = min(largura_img, x1 + w_int)
    y2 = min(altura_img, y1 + h_int)

    pedra_recortada = img_rot[y1:y2, x1:x2]

    if pedra_recortada.size == 0:
        return 0, 0, False, 0.0

    # Verifica se tem conteúdo (zero|zero real vs pedra branca pura)
    contornos_total, _ = cv2.findContours(pedra_recortada, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

    # ----------------------------------------------------------------
    # LOCALIZAR A FENDA (divisor real entre as metades)
    # ----------------------------------------------------------------
    meio, zero_local = _encontrar_fenda(pedra_recortada)

    metade_cima = pedra_recortada[0:meio, :]
    metade_baixo = pedra_recortada[meio:, :]

    if debug_ativo('Pn'):
        cv2.imshow("Medade da Pedra", metade_cima)
        cv2.imshow("Medade da Pedra 2", metade_baixo)
        # print(f"Ratio do traço: {ratio}")
        cv2.waitKey(0)

    med_ar = 0.0
    count = 0
    med_area_bruta = 0.0

    for c in contornos_total:
        area = cv2.contourArea(c)
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))
        if circularidade >= 0.5:
            med_ar += area
            count += 1

    if med_ar > 0:
        med_area_bruta = med_ar / count

    pts_cima, med_ar1 = contar_bolinhas(med_area_bruta, metade_cima)
    pts_baixo, med_ar2 = contar_bolinhas(med_area_bruta, metade_baixo)

    med_area = 0.0
    if med_ar1 > 0.0 or med_ar2 > 0.0:
        med_area = abs((abs(med_ar1) + abs(med_ar2)) / 2)

    return pts_cima, pts_baixo, zero_local, med_area

def _encontrar_fenda(pedra_bin):
    h_total = pedra_bin.shape[0]
    w_total = pedra_bin.shape[1]

    margem = h_total // 4
    zona = pedra_bin[margem: h_total - margem, :]

    # SEM inverter — procura contornos brancos na zona central
    # A fenda deixa bordas brancas finas acima e abaixo dela
    # que formam retângulos longos e finos (ratio alto)
    contornos, _ = cv2.findContours(zona, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    melhor_ratio = 0.0
    melhor_y = None
    largura_minima = w_total * 0.3
    zero_local = False

    for c in contornos:
        x, y, cw, ch = cv2.boundingRect(c)
        if ch == 0 or cw == 0:
            continue

        # Só interessa retângulos MAIS LARGOS que altos (fenda = comprida e fina)
        if cw <= ch:
            continue

        ratio = cw / ch
        # print(f"Valor de cw: {cw}, Valor de ch: {ch}, Ratio = {ratio}")

        if ratio > melhor_ratio and cw >= largura_minima:
            melhor_ratio = ratio
            melhor_y = margem + y + ch // 2
        if ratio > 3.0:
            zero_local = True

    if melhor_y is None:
        return h_total // 2, zero_local

    return melhor_y, zero_local

def contar_bolinhas(med_bruta, metade):
    """
    Conta bolinhas numa metade. Recebe med_bruta para
    poder calibrar area_ponto dinamicamente.
    """
    contornos, _ = cv2.findContours(metade, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    pontos = 0
    # fator_area = args.zoom ** 2
    # point_area = int(CONFIGS['area_ponto'] * fator_area)
    # point_area = CONFIGS['area_ponto']
    med_area = 0.0

    for c in contornos:
        area = cv2.contourArea(c)
        # if point_area * 0.4 < area < point_area * 2.5:
        perimetro = cv2.arcLength(c, True)
        if perimetro == 0:
            continue
        circularidade = 4 * np.pi * (area / (perimetro * perimetro))
        if debug_ativo('Pn'):
            print(f"Circularidade: {circularidade}  --- Area: {area}  --- Area media: {med_bruta}")
        # if (med_bruta * 1.5) > area > (med_bruta * 0.5):
        if circularidade >= 0.5 and (med_bruta * 2.2) >= area >= (med_bruta * 0.1) :   # levemente mais permissivo pós INTER_NEAREST
            med_area += area
            pontos += 1

    if med_area > 0.0 and pontos > 0:
        med_area = abs(med_area / pontos)
    return min(pontos, 6), med_area


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
    if debug_ativo('Vl'):
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
