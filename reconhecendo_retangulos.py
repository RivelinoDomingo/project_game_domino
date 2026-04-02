import cv2
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser(description='Processa imagens')
parser.add_argument('imagem', help='Caminho para o arquivo de imagem')
parser.add_argument('-z', '--zoom', type=float, help='Valor float para nivel de zoom da imagem, padrão 0.8')
parser.add_argument('-p', '--proximidade', type=int, help='Distancia minima entre as pedras padrão 30')
args = parser.parse_args()

zoom_factor = args.zoom
DISTANCIA_MINIMA = args.proximidade

if args.proximidade is None:
    DISTANCIA_MINIMA = 37

if zoom_factor is None:
        zoom_factor = 0.8


def pipeline_blackhat(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Erro: Imagem não encontrada.")
        return
    # Área de zoom
    img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel_bh = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 12))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_bh)

    _, mask_bh = cv2.threshold(blackhat, 100, 255, cv2.THRESH_BINARY)
    kernel_close = np.ones((2,2), np.uint8)
    mask_soldada = cv2.morphologyEx(mask_bh, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask_soldada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    out = img.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 300:
            continue

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (w_box, h_box), angle = rect

        if w_box == 0 or h_box == 0:
            continue

        linha_comprimento = max(w_box, h_box)
        linha_espessura = min(w_box, h_box)
        ratio = linha_comprimento / linha_espessura

        # --- A NOVA BLINDAGEM DE TAMANHO ---
        # A sua pedra projetada tem 30px de largura.
        # O traço costuma ter entre 15px e 26px.
        # Vamos barrar qualquer coisa que seja maior que a própria pedra!
        # --- REFINO DO TRAÇO GORDINHO ---
        LIMITE_MAX_TRACO = 32
        LIMITE_MIN_TRACO = 12
        LIMITE_MAX_ESPESSURA = 5

        # Adicionamos os limites de comprimento no IF principal
        if ratio > 2.0 and (LIMITE_MIN_TRACO <= linha_comprimento <= LIMITE_MAX_TRACO) and (1 <= linha_espessura <= LIMITE_MAX_ESPESSURA):

            if w_box > h_box:
                rect_pedra = ((cx, cy), (30, 61), angle)
            else:
                rect_pedra = ((cx, cy), (61, 30), angle)
            # --- A SUA IDEIA APLICADA AQUI ---
            # 1. Criar uma máscara preta do mesmo tamanho da imagem
            mask_box = np.zeros(gray.shape, dtype=np.uint8)

            # 2. Desenhar a nossa caixa verde preenchida de branco nessa máscara
            box_pedra_pts = np.int32(cv2.boxPoints(rect_pedra))
            cv2.fillPoly(mask_box, [box_pedra_pts], 255)

            # 3. Calcular a média de brilho da imagem original SOMENTE DENTRO da caixa
            # Retorna um valor de 0 (preto absoluto) a 255 (branco absoluto)
            brilho_medio = cv2.mean(gray, mask=mask_box)[0]

            # 4. Só aceitamos se a caixa for predominantemente clara (dominó)
            # Você pode ajustar esse valor (80 a 110) dependendo da sua iluminação
            if brilho_medio > 90:
                candidatos.append({
                    'rect_pedra': rect_pedra,
                    'rect_traco': rect,
                    'centro': (cx, cy),
                    'brilho': brilho_medio  # Guardamos o brilho para o duelo final!
                })

    # --- DUELO E FILTRO DE DISTÂNCIA ---
    # Ordenamos os candidatos do MAIS CLARO para o MAIS ESCURO.
    # Isso garante que caixas centralizadas perfeitas derrotem as caixas de borda.
    candidatos.sort(key=lambda x: x['brilho'], reverse=True)

    # ==========================================
    # PASSO 1: Remover Duplicatas (Caixas na mesma pedra)
    # ==========================================
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
        soma_pts = pts_cima + pts_baixo
        if soma_pts <= 2:
            if not validar_pedra_lisa(gray, d['rect_pedra']):
                # Se for falso positivo (ex: linha na mesa), pula pro próximo laço!
                print(f"Pedra Rejeitada: {texto}")
                pedras_aprovadas.remove(d)
                continue
        box_traco = np.int32(cv2.boxPoints(d['rect_traco']))
        cv2.drawContours(out, [box_traco], 0, (255, 0, 0), 2)

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
    cv2.imshow("Mascara Limpa (Blackhat)", mask_soldada)
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

def validar_pedra_lisa(gray_img, rect_pedra):
    # 1. Obter os 4 cantos da caixa verde
    box = cv2.boxPoints(rect_pedra)
    box = np.int32(box)

    # Usa a sua função já existente para colocar os pontos na ordem certa
    pts = ordenar_pontos(box)

    # 2. "Esticar" a imagem para um retângulo perfeito (ex: 60x120 pixels)
    largura, altura = 60, 120
    pts_destino = np.array([
        [0, 0],
        [largura - 1, 0],
        [largura - 1, altura - 1],
        [0, altura - 1]
    ], dtype="float32")

    matriz = cv2.getPerspectiveTransform(pts, pts_destino)
    pedra_plana = cv2.warpPerspective(gray_img, matriz, (largura, altura))

    # 3. A Guilhotina: Divide exatamente no meio do traço
    metade_cima = pedra_plana[0:altura//2, 0:largura]
    metade_baixo = pedra_plana[altura//2:altura, 0:largura]

    # 4. Calcula o Brilho e a Textura de cada metade individualmente
    mean_c = cv2.meanStdDev(metade_cima)
    mean_b = cv2.meanStdDev(metade_baixo)

    brilho_c = mean_c[0][0].item()
    brilho_b = mean_b[0][0].item()

    # 5. A NOVA REGRA (Simetria de Brilho)
    diferenca_brilho = abs(brilho_c - brilho_b)

    # ========================================================
    # OS CRITÉRIOS DE APROVAÇÃO:
    # 1. As duas metades têm de ser claras (> 100)
    # 2. A diferença de luz entre elas tem de ser pequena (< 40)
    # ========================================================
    if (brilho_c > 100 and brilho_b > 100) and diferenca_brilho < 30:
        return True
    else:
        # Imprime no terminal o MOTIVO da rejeição para você poder calibrar!
        print(f"👻 Fantasma rejeitado! Brilho (C:{brilho_c:.0f}, B:{brilho_b:.0f}) | Dif: {diferenca_brilho:.0f}")
        return False

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
                if 30 < area < 85:
                    # BLINDAGEM 2: O filtro de formato (Circularidade)
                    perimetro = cv2.arcLength(c, True)
                    if perimetro == 0: continue

                    circularidade = 4 * np.pi * (area / (perimetro * perimetro))

                    # Se for redondo o suficiente (Círculo = 1.0, Quadrado ~0.78)
                    if circularidade > 0.4:
                        pontos += 1

            # BLINDAGEM 3: Trava matemática máxima de um dominó
            return min(pontos, 6)

    pts_cima = contar_bolinhas(metade_cima)
    pts_baixo = contar_bolinhas(metade_baixo)

    # Opcional: mostrar as pedras extraídas para você ver a mágica acontecendo (comente depois)
    # cv2.imshow("Pedra Extraida", warped)
    # cv2.waitKey(0)

    return pts_cima, pts_baixo


# TESTE
# pipeline_blackhat("imagem_recortada.jpeg")
# pipeline_blackhat("imagem.jpeg")
pipeline_blackhat(args.imagem)
