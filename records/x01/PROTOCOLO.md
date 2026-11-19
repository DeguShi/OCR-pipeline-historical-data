# X01 — referência humana diplomática (protocolo, escrito antes de transcrever, 2026-11-18)

**Objetivo.** Uma referência feita por pessoa, lendo a imagem, independente do Gemini e dos sistemas avaliados, para
medir de novo os sistemas congelados (R1·P6, R4·PA, R4·P6, PA de 181 configurações, VLM etc.) sem reajustar nada.

**Páginas.** As 10 de teste (18, 22, 23, 24, 25, 31, 32, 34, 44, 55): nunca entraram em treino, rotulagem nem
seleção. E as 3 reservadas (7, 10, 12), que já têm GT oficial, para comparar as três referências. Nenhuma outra
página: todas as demais já foram usadas em treino ou ajuste. Ordem sugerida: as 10 de teste primeiro.

**O que o anotador vê.** A imagem da página e os recortes de linha da segmentação (só para localizar). Por padrão,
nenhuma leitura de máquina e nenhum texto do Gemini ou do GT oficial (o kit é gerado com `--prefill none --aids
none`). Se for usado o rascunho do Gemini (`--prefill gemini`), isso precisa ser declarado no artigo como
"pós-edição do rascunho do Gemini, linha a linha contra a imagem", porque ancora a referência no professor dos
rótulos.

**Regras de transcrição (diplomática).**
1. Uma linha do arquivo por linha do manuscrito, na ordem da página. Linhas em branco não entram.
2. Grafia como está escrita: sem modernizar (*approvação*, *commissão*, *annos*, *direcção* ficam assim), sem
   corrigir erros do escrivão.
3. Acentos como aparecem na imagem; se não houver acento visível, não acrescentar.
4. Maiúsculas e pontuação como no manuscrito (o WER principal ignora caixa e pontuação, mas o estrito usa).
5. Abreviaturas como escritas, sem expandir (ex.: *Snr.*, *Dr.*). Letras sobrescritas: escrever na linha
   (*1.º*, *2.ª*); se houver ponto antes do sobrescrito na imagem, escrever o ponto (*1.º*); se não houver, não
   (*1º*). Este é o caso que mais pesa no WER (Seção do gabarito), então decidir olhando a imagem, caso a caso.
6. Palavra partida no fim da linha: primeira parte seguida de `=` e a continuação no início da linha seguinte
   (mesma convenção das referências atuais; o avaliador junta).
7. Leitura incerta: escrever a melhor leitura seguida de `[?]` (ex.: *Moraes[?]*). Ilegível: `[...]`.
8. Riscado: omitir. Entrelinha/acréscimo do escrivão: inserir no lugar indicado.
9. Cabeçalhos, números de página e carimbos: transcrever se forem escrita do livro; ignorar carimbos de arquivo.

**Registro.** Para cada página, anotar o tempo gasto (minutos) e as dúvidas em `notas.txt` na pasta da página.
Depois de entregue, a referência é congelada (sha256 de cada arquivo) antes de qualquer pontuação; nenhum sistema é
reajustado com ela.

**Uso previsto.** WER/CER normalizados e estritos de todos os sistemas congelados contra a referência humana; teste
por página (permutação e bootstrap de páginas) para R1·P6 contra R4·PA e contra o PA de 181 configurações; e a
concordância das três referências nas reservadas. Se houver uma segunda pessoa, ela revisa as divergências entre a
transcrição e as duas referências existentes (arbitragem), sem acesso às saídas dos sistemas.
