import re
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple, Dict, Optional

@dataclass
class Issue:
    line_index: int
    description: str
    severity: str

class CheckerModule:
    """
    检查与纠正模块基类
    支持检查并可选地执行自动修复逻辑
    """
    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        raise NotImplementedError

class HeaderChecker(CheckerModule):
    """
    头部信息检查与补全模块
    将必须的字段及其默认值作为配置传入
    """
    def __init__(self, required_headers_with_defaults: Dict[str, str]):
        self.required_headers = required_headers_with_defaults

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)
        found_headers = set()

        for line in lines:
            if len(line) >= 2 and line[1] == ':':
                found_headers.add(line[0])

        for header, default_val in self.required_headers.items():
            if header not in found_headers:
                issues.append(Issue(line_index=-1, description=f"缺失必要的头部字段 {header}", severity="error"))
                if auto_fix:
                    # 自动修复逻辑：在文件开头插入缺失的头部及其默认值
                    modified_lines.insert(0, f"{header}:{default_val}")

        return issues, modified_lines

class ClefAutoSelector(CheckerModule):
    """
    自动判断音区并选择合适的谱号 (treble / bass)。

    依据 ABC standard v2.1 §4.1 (Pitch) 与 §4.6 (Clefs and transposition):
      - 大写 C..B  对应中央 C 起的一个八度  (C = MIDI 60)
      - 小写 c..b  对应再上一个八度        (c = MIDI 72)
      - 每个 ','  降低一个八度,  每个 "'" 升高一个八度
      - clef 可写在 K: 或 V: 行, 形如 'clef=bass' 或裸名 'bass'

    策略:
      1. 解析每个声部 (V:) 的全部音符为 MIDI 数值
      2. 统计落在 treble / bass 各自舒适音域之外 (即需要超过若干加线)
         的音符数量
      3. 若另一个谱号能显著减少加线越界数量, 则建议 / 自动替换
         声部的默认谱号 (V: / K: 行)
      4. 进一步按小节扫描音区, 若出现持续若干小节的音区临时切换,
         则在相应小节前插入 inline clef 字段 '[K:clef=xxx]'
         (§3.1 inline field + §4.6 clef)
    """

    # ABC 音名 -> MIDI (基准八度)
    BASE_PITCHES = {
        'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
        'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83,
    }

    # 谱号舒适音区 (允许大致 ledger_limit 条加线之内)
    # treble: 五线谱内 E4(64) - F5(77); 加 2 加线后大约 A3(57) - C6(84)
    # bass:   五线谱内 G2(43) - A3(57); 加 2 加线后大约 E2(40) - E4(64)
    CLEF_RANGES = {
        'treble': (55, 84),
        'bass':   (40, 64),
    }

    KNOWN_CLEFS = ('treble', 'bass', 'alto', 'tenor', 'perc', 'none')

    NOTE_RE = re.compile(r"(\^\^|__|\^|_|=)?([A-Ga-g])([,']*)")

    def __init__(self, ledger_threshold_ratio: float = 0.25,
                 min_improvement: int = 1,
                 min_run_length: int = 2):
        """
        ledger_threshold_ratio:
            当前谱号下越界音符占比超过该值才考虑换谱号
        min_improvement:
            候选谱号必须比当前谱号至少减少这么多越界音符才会被采用
        min_run_length:
            小节级临时切换时, 偏好新谱号的连续小节数必须达到该值,
            否则认为是偶发而不插入 inline clef 标记
        """
        self.ledger_threshold_ratio = ledger_threshold_ratio
        self.min_improvement = min_improvement
        self.min_run_length = min_run_length

    # ---------- 解析辅助 ----------

    def _strip_non_music(self, line: str) -> str:
        """剔除非音符 token, 防止误把和弦符号 / 注释里的字母当成音符"""
        line = re.sub(r'%.*$', '', line)               # 行注释
        line = re.sub(r'"[^"]*"', '', line)            # 和弦符号 / 注释字符串
        line = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', line)  # inline field
        line = re.sub(r'!\w+!', '', line)              # !decoration!
        line = re.sub(r'\+[\w]+\+', '', line)          # +decoration+
        line = re.sub(r'\{[^}]*\}', '', line)          # 装饰音
        return line

    def _note_to_midi(self, match) -> int:
        letter = match.group(2)
        octave_marks = match.group(3) or ''
        midi = self.BASE_PITCHES[letter]
        midi += 12 * octave_marks.count("'")
        midi -= 12 * octave_marks.count(',')
        return midi

    def _extract_pitches(self, music_lines: List[str]) -> List[int]:
        pitches = []
        for line in music_lines:
            cleaned = self._strip_non_music(line)
            for m in self.NOTE_RE.finditer(cleaned):
                pitches.append(self._note_to_midi(m))
        return pitches

    def _out_of_range(self, pitches: List[int], clef: str) -> int:
        low, high = self.CLEF_RANGES[clef]
        return sum(1 for p in pitches if p < low or p > high)

    def _suggest_clef(self, pitches: List[int], current: str) -> Optional[str]:
        if not pitches:
            return None
        if current not in self.CLEF_RANGES:
            current = 'treble'
        cur_out = self._out_of_range(pitches, current)
        # 越界比例不大就保持现状
        if cur_out / len(pitches) < self.ledger_threshold_ratio:
            return None
        # 在 treble / bass 中挑越界最少的
        best = min(self.CLEF_RANGES.keys(), key=lambda c: self._out_of_range(pitches, c))
        if best == current:
            return None
        if cur_out - self._out_of_range(pitches, best) < self.min_improvement:
            return None
        return best

    # ---------- clef 字段读写 ----------

    def _extract_clef(self, line: str) -> Optional[str]:
        m = re.search(r'clef\s*=\s*([A-Za-z0-9+\-]+)', line)
        if m:
            name = m.group(1)
            for c in self.KNOWN_CLEFS:
                if name.startswith(c):
                    return c
            return None
        # 裸的谱号名 (例如 'K:C bass' 或 'V:1 treble')
        head = re.match(r'\s*[KV]:\s*(.*)', line)
        if head:
            for tok in head.group(1).split():
                if tok in self.KNOWN_CLEFS:
                    return tok
        return None

    def _set_clef(self, line: str, clef: str) -> str:
        if re.search(r'clef\s*=\s*[A-Za-z0-9+\-]+', line):
            return re.sub(r'clef\s*=\s*[A-Za-z0-9+\-]+', f'clef={clef}', line)
        for c in ('treble', 'bass', 'alto', 'tenor'):
            if re.search(rf'(?<![A-Za-z]){c}(?![A-Za-z])', line):
                return re.sub(rf'(?<![A-Za-z]){c}(?![A-Za-z])', clef, line, count=1)
        return line.rstrip() + f' clef={clef}'

    # ---------- 主流程 ----------

    @staticmethod
    def _is_header_line(line: str) -> bool:
        s = line.lstrip()
        return len(s) >= 2 and s[1] == ':' and s[0].isalpha()

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues: List[Issue] = []
        modified_lines = list(lines)

        # 找到第一个 K: 行 — ABC 规定 K: 之后才是乐谱主体
        body_start = None
        k_line_idx = None
        for i, l in enumerate(lines):
            if l.lstrip().startswith('K:'):
                k_line_idx = i
                body_start = i + 1
                break
        if body_start is None:
            return issues, modified_lines

        global_clef = self._extract_clef(lines[k_line_idx]) or 'treble'

        # 把 body 按 V: 切分成段; 没出现 V: 则整体当作一段
        segments: List[Tuple[Optional[int], List[int]]] = []
        current_voice_idx: Optional[int] = None
        current_music: List[int] = []

        def flush():
            if current_voice_idx is not None or current_music:
                segments.append((current_voice_idx, list(current_music)))

        for i in range(body_start, len(lines)):
            line = lines[i]
            if line.lstrip().startswith('V:'):
                flush()
                current_voice_idx = i
                current_music = []
            elif line.strip() and not self._is_header_line(line):
                current_music.append(i)
        flush()

        if not segments:
            return issues, modified_lines

        for voice_idx, music_idxs in segments:
            music_text = [lines[i] for i in music_idxs]
            pitches = self._extract_pitches(music_text)
            if not pitches:
                continue

            if voice_idx is not None:
                current_clef = self._extract_clef(lines[voice_idx]) or global_clef
                target_idx = voice_idx
            else:
                current_clef = global_clef
                target_idx = k_line_idx

            # --- 声部级: 整体重选谱号 ---
            suggested = self._suggest_clef(pitches, current_clef)
            effective_clef = current_clef
            if suggested is not None:
                cur_out = self._out_of_range(pitches, current_clef)
                new_out = self._out_of_range(pitches, suggested)
                desc = (f"音区更适合 {suggested} 谱号 "
                        f"(当前 {current_clef}: 越界 {cur_out}/{len(pitches)}, "
                        f"建议后越界 {new_out}/{len(pitches)})")
                issues.append(Issue(line_index=target_idx, description=desc, severity="info"))
                if auto_fix:
                    modified_lines[target_idx] = self._set_clef(modified_lines[target_idx], suggested)
                effective_clef = suggested

            # --- 小节级: 检测临时音区变化, 插入 inline clef ---
            self._process_inline_clefs(
                lines=lines,
                modified_lines=modified_lines,
                music_idxs=music_idxs,
                starting_clef=effective_clef,
                auto_fix=auto_fix,
                issues=issues,
            )

        return issues, modified_lines

    # ---------- 小节级 inline clef 处理 ----------

    def _best_clef_for(self, pitches: List[int]) -> Optional[str]:
        if not pitches:
            return None
        return min(self.CLEF_RANGES.keys(),
                   key=lambda c: self._out_of_range(pitches, c))

    def _process_inline_clefs(self,
                              lines: List[str],
                              modified_lines: List[str],
                              music_idxs: List[int],
                              starting_clef: str,
                              auto_fix: bool,
                              issues: List[Issue]) -> None:
        """扫描声部内各小节的音区, 必要时插入 [K:clef=xxx] inline 标记"""

        # 按单个 '|' 切分每一行, 偶数索引是小节内容, 奇数索引是小节线
        line_chunks: Dict[int, List[str]] = {}
        measures: List[Dict] = []

        for li in music_idxs:
            parts = re.split(r'(\|)', lines[li])
            line_chunks[li] = parts
            for ci in range(0, len(parts), 2):
                content = parts[ci]
                cleaned = self._strip_non_music(content)
                pitches = [self._note_to_midi(m)
                           for m in self.NOTE_RE.finditer(cleaned)]
                # 纯空白 / 纯小节线装饰 (例如开头空 chunk) 跳过
                if not cleaned.strip() and not pitches:
                    continue
                measures.append({
                    'line_idx': li,
                    'chunk_idx': ci,
                    'pitches': pitches,
                })

        if len(measures) < self.min_run_length + 1:
            return

        m_clefs = [self._best_clef_for(m['pitches']) for m in measures]

        if starting_clef not in self.CLEF_RANGES:
            starting_clef = 'treble'
        active = starting_clef

        inserts: List[Tuple[int, int, str]] = []  # (line_idx, chunk_idx, clef)

        i = 0
        while i < len(measures):
            c = m_clefs[i]
            # 无音符 / 已经是当前谱号 -> 跳过
            if c is None or c == active:
                i += 1
                continue
            # 向前看: 统计连续偏好 c 的小节 (允许中间夹空小节)
            j = i
            run = 0
            while j < len(measures) and m_clefs[j] in (c, None):
                if m_clefs[j] == c:
                    run += 1
                j += 1
            if run < self.min_run_length:
                i = j
                continue
            # 验证切换确有改善
            run_pitches: List[int] = []
            for k in range(i, j):
                run_pitches.extend(measures[k]['pitches'])
            if (self._out_of_range(run_pitches, active)
                    - self._out_of_range(run_pitches, c)) < self.min_improvement:
                i = j
                continue
            inserts.append((measures[i]['line_idx'],
                            measures[i]['chunk_idx'], c))
            active = c
            i = j

        if not inserts:
            return

        for li, _, c in inserts:
            issues.append(Issue(
                line_index=li,
                description=f"音区临时切换, 建议在该小节前插入 inline 谱号 [K:clef={c}]",
                severity="info",
            ))

        if not auto_fix:
            return

        # 应用插入 (从后往前以保持 chunk 索引稳定)
        inserts_sorted = sorted(inserts,
                                key=lambda x: (x[0], x[1]),
                                reverse=True)
        for li, ci, c in inserts_sorted:
            chunks = line_chunks[li]
            original = chunks[ci]
            stripped = original.lstrip()
            leading = original[:len(original) - len(stripped)]
            marker = f'[K:clef={c}]'
            # 保留原有前导空白, 紧接 marker, 再接原内容
            chunks[ci] = f'{leading}{marker}{stripped}' if stripped else f'{leading}{marker}'

        for li, chunks in line_chunks.items():
            modified_lines[li] = ''.join(chunks)


class VoiceBarCountChecker(CheckerModule):
    """
    检查各声部 (V:) 的小节数是否一致。
    以最多小节数的声部为基准，对声部数不足的声部报告 warning。
    auto_fix=True 时，在不足的声部末尾补齐空白小节，
    填充内容为 x 音符，数量由 M: 与 L: 计算得出。
    """

    # 匹配所有合法的小节线变体，统一归一为 '|'
    _BAR_RE = re.compile(r':\|:|:\||\|\||\|\]|\[?\|:?')

    def _count_measures(self, music_lines: List[str]) -> int:
        content = ' '.join(music_lines)
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)   # 行注释
        content = re.sub(r'"[^"]*"', '', content)                    # 和弦/注释字符串
        content = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', content)        # inline field
        content = self._BAR_RE.sub('|', content)                     # 归一化小节线
        parts = content.split('|')
        return sum(1 for p in parts if p.strip())

    def _notes_per_measure(self, lines: List[str]) -> int:
        """
        根据 M: 与 L: 计算每小节内含多少个 L 时值单位。
        公式: (M_分子/M_分母) / (L_分子/L_分母) = M_分子 * L_分母 / (M_分母 * L_分子)
        例: M:4/4, L:1/4 -> (4*4)/(4*1) = 4
        """
        meter = unit = None
        for line in lines:
            if re.match(r'\s*M:\s*\d', line):
                m = re.match(r'\s*M:\s*(\d+)/(\d+)', line)
                if m:
                    meter = (int(m.group(1)), int(m.group(2)))
            if re.match(r'\s*L:\s*\d', line):
                m = re.match(r'\s*L:\s*(\d+)/(\d+)', line)
                if m:
                    unit = (int(m.group(1)), int(m.group(2)))
        if meter is None or unit is None:
            return 4  # 默认回退
        numerator = meter[0] * unit[1]
        denominator = meter[1] * unit[0]
        return max(1, round(numerator / denominator))

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        # 找到 K: 行，乐谱主体从其后一行开始
        body_start = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                body_start = i + 1
                break
        if body_start is None:
            return issues, modified_lines

        notes_per_measure = self._notes_per_measure(lines[:body_start])

        # 按 V: 切分声部
        # voices: vid -> (v_line_idx, last_music_line_idx, music_lines)
        voices: Dict[str, Tuple[int, int, List[str]]] = {}
        current_id = '__default__'
        current_v_idx = body_start
        current_last_music_idx = body_start
        current_music: List[str] = []

        def _flush():
            if current_music:
                voices[current_id] = (current_v_idx, current_last_music_idx, list(current_music))

        for i in range(body_start, len(lines)):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith('V:'):
                _flush()
                current_music.clear()
                voice_body = stripped[2:].strip()
                current_id = voice_body.split()[0] if voice_body else str(i)
                current_v_idx = i
                current_last_music_idx = i
            elif stripped and not (len(stripped) >= 2 and stripped[1] == ':' and stripped[0].isalpha()):
                current_music.append(line)
                current_last_music_idx = i

        _flush()
        voices.pop('__default__', None)

        if len(voices) < 2:
            return issues, modified_lines

        counts = {vid: (v_idx, last_idx, self._count_measures(music))
                  for vid, (v_idx, last_idx, music) in voices.items()}

        all_counts = [c for _, _, c in counts.values()]
        if len(set(all_counts)) == 1:
            return issues, modified_lines

        max_count = max(all_counts)

        for vid, (v_idx, last_idx, count) in counts.items():
            if count >= max_count:
                continue

            missing = max_count - count
            issues.append(Issue(
                line_index=v_idx,
                description=(f"声部 V:{vid} 共 {count} 个小节，"
                             f"少于最长声部的 {max_count} 个（缺 {missing} 小节）"),
                severity="warning",
            ))

            if auto_fix:
                measure_content = f'x{notes_per_measure}'
                fill_parts = ' | '.join([measure_content] * missing)
                last_line = modified_lines[last_idx].rstrip()
                if last_line.endswith('|'):
                    modified_lines[last_idx] = last_line + ' ' + fill_parts + ' |'
                else:
                    modified_lines[last_idx] = last_line + ' | ' + fill_parts + ' |'

        return issues, modified_lines


class MeasureDurationChecker(CheckerModule):
    """
    检查各声部每个小节的时值总和是否符合 M: 与该声部默认时值 L: 的设定。

    流程：
      1. 调用 VoiceBarCountChecker（纯检查模式）确认各声部小节数一致；
         若不一致则直接返回小节数问题，不继续检查。
      2. 解析头部 M: 与全局 L:（仅作回退）；每个带 V: 的声部在 V: 至首行乐谱之间扫描 L:，
         未写则使用全局 L:。按各声部自身的 L: 计算每小节期望的 L 单位数。
      3. 逐声部、逐小节解析所有音符时值，与期望值比对。

    auto_fix=True 时：
      - 不足时值的小节末尾补 z 休止符；
      - 超出时值的小节从末尾逐音删除直至 ≤ 期望，再按不足条件补休止；
      - 弱起检测：若所有声部第一小节在全音符尺度下均短于拍号且彼此相等，识别为弱起，
        跳过全部第一小节；否则第一小节对齐到「全音符下最长」再换算回各声部 L 单位修复。
    """

    _BAR_RE            = re.compile(r':\|:|:\||\|\||\|\]|\[?\|:?')
    _TUPLET_Q          = {2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2}
    _TUPLET_Q_COMPOUND = {2: 3, 3: 2, 4: 3, 5: 3, 6: 2, 7: 3, 8: 3, 9: 3}

    # ---------- 头部解析 ----------

    def _parse_headers(self, lines: List[str]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        meter = unit = None
        for line in lines:
            s = line.lstrip()
            if s.startswith('M:'):
                val = s[2:].strip()
                if val in ('C', 'c'):        meter = (4, 4)
                elif val in ('C|', 'c|'):    meter = (2, 2)
                elif val.lower() == 'none':  meter = None
                else:
                    m = re.match(r'\(?([\d+]+)\)?/(\d+)', val)
                    if m:
                        num = sum(int(x) for x in m.group(1).split('+'))
                        meter = (num, int(m.group(2)))
            if s.startswith('L:'):
                m = re.match(r'L:\s*(\d+)/(\d+)', s)
                if m:
                    unit = (int(m.group(1)), int(m.group(2)))
        if unit is None and meter is not None:
            ratio = meter[0] / meter[1]
            unit = (1, 16) if ratio < 0.75 else (1, 8)
        return meter, unit

    @staticmethod
    def _parse_l_line(line: str) -> Optional[Tuple[int, int]]:
        s = line.lstrip()
        if not s.startswith('L:'):
            return None
        m = re.match(r'L:\s*(\d+)/(\d+)', s)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _scan_voice_l_unit(
        self,
        lines: List[str],
        v_line_idx: int,
        first_music_line_idx: Optional[int],
        next_v_line_idx: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """
        在 V: 行之后、该声部首行乐谱之前扫描 L:（中间可含 %%、空行、注释等）。
        若未找到则返回 None，由调用方回退到全局 L:。
        """
        start = v_line_idx + 1
        end = len(lines)
        if next_v_line_idx is not None:
            end = min(end, next_v_line_idx)
        if first_music_line_idx is not None:
            end = min(end, first_music_line_idx)
        for j in range(start, end):
            raw = lines[j]
            s = raw.lstrip()
            if not s:
                continue
            if s.startswith('%'):
                continue
            if s.startswith('%%'):
                continue
            if s.startswith('V:'):
                break
            parsed = self._parse_l_line(s)
            if parsed is not None:
                return parsed
        return None

    def _is_compound(self, mn: int, md: int) -> bool:
        return md == 8 and mn in (6, 9, 12)

    # ---------- 清理 ----------

    def _clean(self, s: str) -> str:
        s = re.sub(r'%.*$', '', s, flags=re.MULTILINE)
        s = re.sub(r'"[^"]*"', '', s)
        s = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', s)
        s = re.sub(r'\{/?[^}]*\}', '', s)
        s = re.sub(r'![^!]*!', '', s)
        s = re.sub(r'\+[^+]+\+', '', s)
        return s

    # ---------- 长度修饰符解析 ----------

    def _parse_len(self, s: str, pos: int) -> Tuple[Fraction, int]:
        n, start = len(s), pos
        num_str = ''
        while pos < n and s[pos].isdigit():
            num_str += s[pos]; pos += 1
        slash_cnt = 0
        while pos < n and s[pos] == '/':
            slash_cnt += 1; pos += 1
        den_str = ''
        while pos < n and s[pos].isdigit():
            den_str += s[pos]; pos += 1
        if not num_str and not slash_cnt:
            return Fraction(1), start
        num = int(num_str) if num_str else 1
        den = (int(den_str) if den_str else 2 ** slash_cnt) if slash_cnt else 1
        return Fraction(num, den), pos

    # ---------- 单小节时值计算（共用核心） ----------

    def _calc(self, s: str, is_compound: bool, track_pos: bool = False
              ) -> 'Tuple[Optional[Fraction], List[Tuple[int,int,Fraction]]]':
        """
        计算字符串 s 中所有音符时值之和，并可选地返回每个 token 的位置列表。
        track_pos=True 时在原始字符串中追踪位置（用于截断修复）；
        track_pos=False 时先调用 _clean，仅关心总和（用于检查）。
        返回 (total_or_None, tokens)。
        total=None 表示含多小节休止符 Z/X，跳过。
        """
        if not track_pos:
            s = self._clean(s)
        n = len(s)
        total = Fraction(0)
        pos = 0
        tuplet_stack: List[List] = []
        tokens: List[Tuple[int, int, Fraction]] = []
        # 附点节奏状态（< > broken rhythm）
        next_note_mult = Fraction(1)   # 下一个音符的缩放因子
        last_actual_dur = Fraction(0)  # 上一个音符的实际时值（用于追溯修正）

        def _add(start: int, end: int, dur: Fraction):
            nonlocal total, next_note_mult, last_actual_dur
            actual_dur = dur * next_note_mult
            next_note_mult = Fraction(1)
            if tuplet_stack:
                actual_dur *= tuplet_stack[-1][1]
                tuplet_stack[-1][0] -= 1
                if tuplet_stack[-1][0] <= 0:
                    tuplet_stack.pop()
            total += actual_dur
            last_actual_dur = actual_dur
            if track_pos:
                tokens.append((start, end, actual_dur))

        while pos < n:
            c = s[pos]
            if c in ' \t\n~HLMOPSTuv.': pos += 1; continue
            if c in ')-_&\\':           pos += 1; continue

            # 附点节奏 > 和 < (§4.14)
            # n 个 > ：前音符 ×(2^(n+1)−1)/2^n，后音符 ×1/2^n
            # n 个 < ：前音符 ×1/2^n，后音符 ×(2^(n+1)−1)/2^n
            if c in '<>':
                kind = c
                cnt = 1
                pos += 1
                while pos < n and s[pos] == kind:
                    cnt += 1; pos += 1
                n_pow = 2 ** cnt
                if kind == '>':
                    prev_mult = Fraction(2 * n_pow - 1, n_pow)
                    next_note_mult = Fraction(1, n_pow)
                else:
                    prev_mult = Fraction(1, n_pow)
                    next_note_mult = Fraction(2 * n_pow - 1, n_pow)
                # 追溯修正上一个音符
                if last_actual_dur:
                    delta = last_actual_dur * (prev_mult - 1)
                    total += delta
                    last_actual_dur += delta
                    if track_pos and tokens:
                        s_t, e_t, old = tokens[-1]
                        tokens[-1] = (s_t, e_t, old + delta)
                continue

            # 跳过注释（仅 track_pos 模式，clean 模式已被 _clean 去掉）
            if c == '%':
                while pos < n and s[pos] != '\n': pos += 1
                continue
            # 跳过和弦符号、装饰音、inline 字段（track_pos 模式）
            if c == '"':
                pos += 1
                while pos < n and s[pos] != '"': pos += 1
                if pos < n: pos += 1
                continue
            if c == '{':
                pos += 1
                while pos < n and s[pos] != '}': pos += 1
                if pos < n: pos += 1
                continue
            if c == '!':
                pos += 1
                while pos < n and s[pos] != '!': pos += 1
                if pos < n: pos += 1
                continue
            if c == '+':
                pos += 1
                while pos < n and s[pos] != '+': pos += 1
                if pos < n: pos += 1
                continue
            if (c == '[' and pos + 1 < n and s[pos+1].isalpha()
                    and pos + 2 < n and s[pos+2] == ':'):
                pos += 1
                while pos < n and s[pos] != ']': pos += 1
                if pos < n: pos += 1
                continue

            # 连音符 (§4.13)：(p 或 (p:q 或 (p:q:r 或 (p::r
            # (p 后没有数字 → 是连弧线（slur），无时值效果，直接跳过
            if c == '(':
                pos += 1
                p_str = ''
                while pos < n and s[pos].isdigit(): p_str += s[pos]; pos += 1
                if not p_str: continue          # slur，无效果
                p = int(p_str); q = r = None
                if pos < n and s[pos] == ':':
                    pos += 1
                    q_str = ''
                    while pos < n and s[pos].isdigit(): q_str += s[pos]; pos += 1
                    q = int(q_str) if q_str else None
                    if pos < n and s[pos] == ':':
                        pos += 1
                        r_str = ''
                        while pos < n and s[pos].isdigit(): r_str += s[pos]; pos += 1
                        r = int(r_str) if r_str else None
                tbl = self._TUPLET_Q_COMPOUND if is_compound else self._TUPLET_Q
                if q is None: q = tbl.get(p, 2)
                if r is None: r = p
                if p > 0:
                    tuplet_stack.append([r, Fraction(q, p)])
                continue

            # 和弦
            if c == '[':
                tk_start = pos; pos += 1
                if pos < n and s[pos].isdigit(): pos += 1; continue  # [1 [2
                while pos < n and s[pos] in '^_=': pos += 1
                internal = Fraction(1)
                if pos < n and s[pos] in 'ABCDEFGabcdefgzx':
                    pos += 1; internal, pos = self._parse_len(s, pos)
                while pos < n and s[pos] != ']': pos += 1
                if pos < n: pos += 1
                ext, np2 = self._parse_len(s, pos)
                chord_len = ext if np2 > pos else internal
                pos = np2 if np2 > pos else pos
                _add(tk_start, pos, chord_len)
                continue

            # 临时变音 + 音符/休止符
            if c in '^_=ABCDEFGabcdefgzxZX':
                tk_start = pos
                while pos < n and s[pos] in '^_=': pos += 1
                if pos >= n or s[pos] not in 'ABCDEFGabcdefgzxZX':
                    continue
                nc = s[pos]; pos += 1
                if nc in 'ZX':
                    while pos < n and s[pos].isdigit(): pos += 1
                    return None, []     # 多小节休止符
                # 跳过八度标记 , 和 '（不影响时值）
                while pos < n and s[pos] in ",'":
                    pos += 1
                note_len, pos = self._parse_len(s, pos)
                _add(tk_start, pos, note_len)
                continue

            pos += 1

        return total, tokens

    def _measure_duration(self, measure_str: str, is_compound: bool) -> Optional[Fraction]:
        total, _ = self._calc(measure_str, is_compound, track_pos=False)
        return total

    # ---------- 时值修复辅助 ----------

    @staticmethod
    def _rest_str(f: Fraction) -> str:
        """Fraction → ABC z 休止符写法"""
        num, den = f.numerator, f.denominator
        if den == 1:   return 'z' if num == 1 else f'z{num}'
        if num == 1 and den == 2: return 'z/'
        if num == 1:   return f'z/{den}'
        return f'z{num}/{den}'

    def _fix_measure(self, measure: str, expected: Fraction, is_compound: bool) -> str:
        """
        修复单小节时值：
          不足 → 末尾补 z 休止；
          超出 → 从末尾逐 token 删除直至 ≤ expected，再按不足条件补休止。
        """
        actual = self._measure_duration(measure, is_compound)
        if actual is None or actual == expected:
            return measure

        if actual < expected:
            return measure.rstrip() + ' ' + self._rest_str(expected - actual)

        # 超出：追踪 token 位置，找截断点
        _, tokens = self._calc(measure, is_compound, track_pos=True)
        if not tokens:
            return measure

        cumsum = Fraction(0)
        cut_pos = 0
        for tk_start, tk_end, dur in tokens:
            if cumsum + dur <= expected:
                cumsum += dur
                cut_pos = tk_end
            else:
                break

        fixed = measure[:cut_pos]
        if cumsum < expected:
            fixed = fixed.rstrip() + ' ' + self._rest_str(expected - cumsum)
        return fixed

    def _fix_content(self, content: str, expected: Fraction,
                     is_compound: bool, first_expected: Optional[Fraction]) -> str:
        """
        修复声部内容字符串中所有小节的时值。
        first_expected=None  → 跳过第一小节（弱起，不补足）；
        first_expected=Fraction → 第一小节按该值修复，其余按 expected 修复。
        """
        bar_spans = list(self._BAR_RE.finditer(content))
        segments: List[Tuple[str, str]] = []
        start = 0
        for m in bar_spans:
            segments.append((content[start:m.start()], m.group()))
            start = m.end()
        trailing = content[start:]
        if trailing.strip():
            segments.append((trailing, ''))

        result = []
        for i, (measure, bar) in enumerate(segments):
            if not measure.strip():
                result.append(measure + bar)
                continue
            if i == 0:
                if first_expected is None:
                    result.append(measure + bar)   # 弱起：原样保留
                    continue
                exp = first_expected
            else:
                exp = expected
            fixed = self._fix_measure(measure, exp, is_compound)
            sep = ' ' if (fixed and bar and not fixed.endswith(' ')) else ''
            result.append(fixed + sep + bar)
        return ''.join(result)

    # ---------- 主流程 ----------

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []

        # Step 1: 前置小节数一致性检查
        bar_issues, _ = VoiceBarCountChecker().process(lines, auto_fix=False)
        if bar_issues:
            for bi in bar_issues:
                bi.description = '[前置检查] 小节数不一致，已跳过时值检查：' + bi.description
            return bar_issues, list(lines)

        # Step 2: 定位 K:
        body_start = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                body_start = i + 1; break
        if body_start is None:
            return issues, list(lines)

        # Step 3: 解析 M: 与全局 L:（全局 L 仅作声部未写 L: 时的回退）
        meter, global_unit = self._parse_headers(lines[:body_start])
        if meter is None or global_unit is None:
            return issues, list(lines)
        bar_semibreves = Fraction(meter[0], meter[1])
        is_cmpd = self._is_compound(meter[0], meter[1])

        # Step 4: 切分声部，记录行索引
        voices: Dict[str, Tuple[int, List[int]]] = {}
        cur_id, cur_v_idx = '__default__', body_start
        cur_idxs: List[int] = []

        def _flush():
            if cur_idxs:
                voices[cur_id] = (cur_v_idx, list(cur_idxs))

        def _is_hdr(s: str) -> bool:
            return len(s) >= 2 and s[1] == ':' and s[0].isalpha()

        for i in range(body_start, len(lines)):
            s = lines[i].lstrip()
            if s.startswith('V:'):
                _flush(); cur_idxs = []
                vb = s[2:].strip()
                cur_id = vb.split()[0] if vb else str(i)
                cur_v_idx = i
            elif s and not _is_hdr(s) and not s.startswith('%'):
                cur_idxs.append(i)
        _flush()
        voices.pop('__default__', None)

        if not voices:
            body_idxs = [i for i in range(body_start, len(lines))
                         if lines[i].lstrip() and not _is_hdr(lines[i].lstrip())]
            if body_idxs:
                voices['1'] = (body_start, body_idxs)

        # 各声部默认时值 L:（V: 块内扫描，否则用全局）
        v_ordered = sorted(voices.items(), key=lambda kv: kv[1][0])
        voice_units: Dict[str, Tuple[int, int]] = {}
        expected_by_vid: Dict[str, Fraction] = {}
        for k, (vid, (v_idx, idxs)) in enumerate(v_ordered):
            next_v_line = v_ordered[k + 1][1][0] if k + 1 < len(v_ordered) else None
            first_music = idxs[0] if idxs else None
            scanned = self._scan_voice_l_unit(lines, v_idx, first_music, next_v_line)
            u = scanned if scanned is not None else global_unit
            voice_units[vid] = u
            expected_by_vid[vid] = bar_semibreves / Fraction(u[0], u[1])

        modified_lines = list(lines)

        # ---------- 弱起检测（第一小节时值换算到全音符比例后跨声部比较）----------
        def _first_measure_dur(idxs: List[int]) -> Optional[Fraction]:
            if not idxs: return None
            content = ' '.join(lines[i] for i in idxs)
            content = re.sub(r'"[^"]*"', '', re.sub(r'\[[A-Za-z]:[^\]]*\]', '', content))
            segs = [s for s in self._BAR_RE.split(content) if s.strip()]
            return self._measure_duration(segs[0], is_cmpd) if segs else None

        first_durs: Dict[str, Optional[Fraction]] = {
            vid: _first_measure_dur(idxs)
            for vid, (_, idxs) in voices.items()
        }

        first_semibreves: Dict[str, Fraction] = {}
        for vid, d in first_durs.items():
            if d is None:
                continue
            u = voice_units[vid]
            first_semibreves[vid] = d * Fraction(u[0], u[1])

        voices_with_first = [vid for vid in voices if first_durs.get(vid) is not None]
        is_global_pickup = (
            len(voices) > 0
            and len(voices_with_first) == len(voices)
            and len(first_semibreves) == len(voices)
            and len(set(first_semibreves.values())) == 1
            and next(iter(first_semibreves.values())) != bar_semibreves
        )

        # 非弱起时：第一小节以「全音符下最长者」为基准，再换算回各声部自身 L 单位
        max_first_by_vid: Dict[str, Optional[Fraction]] = {}
        if not is_global_pickup:
            if first_semibreves:
                max_s = max(first_semibreves.values())
            else:
                max_s = bar_semibreves
            for vid in voices:
                u = voice_units[vid]
                max_first_by_vid[vid] = max_s / Fraction(u[0], u[1])
        else:
            for vid in voices:
                max_first_by_vid[vid] = None

        # ---------- 检查 + 修复 ----------
        for vid, (v_idx, idxs) in voices.items():
            if not idxs:
                continue

            content_orig = ' '.join(lines[i] for i in idxs)
            # 预处理（仅用于小节分割和检查，不影响 _fix_content）
            pre = re.sub(r'"[^"]*"', '',
                  re.sub(r'\[[A-Za-z]:[^\]]*\]', '', content_orig))
            measures = [m for m in self._BAR_RE.sub('|', pre).split('|')]
            exp_voice = expected_by_vid[vid]

            for m_num, measure in enumerate(measures, start=1):
                if not measure.strip():
                    continue
                actual = self._measure_duration(measure, is_cmpd)
                if actual is None:
                    continue

                if m_num == 1:
                    if is_global_pickup:
                        continue   # 弱起，跳过检查
                    ref = max_first_by_vid.get(vid)
                    if ref is None or actual == ref:
                        continue
                    hint = f"（本声部 L:{voice_units[vid][0]}/{voice_units[vid][1]} 下对齐至最长小节 {ref} 单位）"
                else:
                    ref = exp_voice
                    if actual == ref:
                        continue
                    hint = f"（本声部 L:{voice_units[vid][0]}/{voice_units[vid][1]}）"

                issues.append(Issue(
                    line_index=v_idx,
                    description=(f"声部 V:{vid} 第 {m_num} 小节时值为 {actual} L单位，"
                                 f"期望 {ref}{hint}"),
                    severity="warning",
                ))

            if auto_fix:
                fixed = self._fix_content(
                    content_orig, exp_voice, is_cmpd, max_first_by_vid.get(vid)
                )
                if fixed != content_orig:
                    modified_lines[idxs[0]] = fixed
                    for idx in idxs[1:]:
                        modified_lines[idx] = ' '

        if is_global_pickup:
            issues.append(Issue(
                line_index=body_start,
                description="所有声部第一小节均不满足拍号且时值相等，已识别为弱起小节，第一小节未修复",
                severity="info",
            ))

        return issues, modified_lines


class TempoEstimator(CheckerModule):
    """
    根据 M:、L: 及音符密度估算速度并建议 Q: 字段。

    算法：
      1. 若已存在 Q: 行则直接跳过。
      2. 解析 M: 得到每拍时值（简单拍号取 1/分母；复合拍号 6/8、9/8、12/8 取 3/8）。
      3. 解析 L: 得到基本音符单位。
      4. 扫描乐谱主体，统计所有音符（含单小节休止符 z）时值，
         计算平均时值（L 单位），再换算为每拍平均音符数。
      5. 按密度映射到建议 BPM（每拍音符越多 → 速度越慢，以保证可演奏性）：
           < 1 个/拍  → Allegro  ≈ 132
           1–2 个/拍  → Moderato ≈ 108
           2–4 个/拍  → Andante  ≈ 80
           ≥ 4 个/拍  → Adagio   ≈ 60
      6. 报告建议（severity=info）；auto_fix=True 时在 K: 行前插入 Q: 行。
    """

    _NOTE_DUR_RE = re.compile(
        r'(?:\^\^|__|[\^_=])?'
        r'(?P<pitch>[A-Ga-gz])'
        r"[,']*"
        r'(?P<num>\d*)(?P<slashes>/*)(?P<den>\d*)'
    )

    # (每拍音符数上限, 建议BPM, 速度术语)
    _DENSITY_MAP = [
        (1.0,           132, "Allegro"),
        (2.0,           108, "Moderato"),
        (4.0,            80, "Andante"),
        (float('inf'),   60, "Adagio"),
    ]

    def _parse_meter(self, header_lines: List[str]) -> Optional[Tuple[int, int]]:
        for line in header_lines:
            s = line.lstrip()
            if not s.startswith('M:'):
                continue
            val = s[2:].strip()
            if val in ('C', 'c'):    return (4, 4)
            if val in ('C|', 'c|'): return (2, 2)
            m = re.match(r'(\d+)/(\d+)', val)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _parse_unit(self, header_lines: List[str]) -> Optional[Tuple[int, int]]:
        for line in header_lines:
            s = line.lstrip()
            if not s.startswith('L:'):
                continue
            m = re.match(r'L:\s*(\d+)/(\d+)', s)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _beat_unit(self, meter: Tuple[int, int]) -> Fraction:
        mn, md = meter
        if md == 8 and mn in (6, 9, 12):
            return Fraction(3, 8)
        return Fraction(1, md)

    def _clean_line(self, line: str) -> str:
        line = re.sub(r'%.*$', '', line)
        line = re.sub(r'"[^"]*"', '', line)
        line = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', line)
        line = re.sub(r'\{[^}]*\}', '', line)
        line = re.sub(r'![^!]*!', '', line)
        line = re.sub(r'\+[^+]+\+', '', line)
        line = re.sub(r'[ZX]\d*', '', line)  # 多小节休止符，不计入密度
        return line

    def _collect_durations(self, music_lines: List[str]) -> List[Fraction]:
        durations = []
        for line in music_lines:
            cleaned = self._clean_line(line)
            for m in self._NOTE_DUR_RE.finditer(cleaned):
                num_s   = m.group('num')
                slashes = m.group('slashes')
                den_s   = m.group('den')
                num = int(num_s) if num_s else 1
                if slashes:
                    den = int(den_s) if den_s else 2 ** len(slashes)
                else:
                    den = 1
                durations.append(Fraction(num, den))
        return durations

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        if any(l.lstrip().startswith('Q:') for l in lines):
            return issues, modified_lines

        k_idx = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                k_idx = i
                break
        if k_idx is None:
            return issues, modified_lines

        meter = self._parse_meter(lines[:k_idx + 1])
        unit  = self._parse_unit(lines[:k_idx + 1])
        if meter is None or unit is None:
            return issues, modified_lines

        music_lines = [
            l for l in lines[k_idx + 1:]
            if l.strip()
            and not (len(l.lstrip()) >= 2 and l.lstrip()[1] == ':' and l.lstrip()[0].isalpha())
        ]
        durations = self._collect_durations(music_lines)
        if not durations:
            return issues, modified_lines

        avg_dur_l  = sum(durations) / len(durations)
        avg_dur_sb = avg_dur_l * Fraction(unit[0], unit[1])

        beat = self._beat_unit(meter)
        notes_per_beat = float(beat / avg_dur_sb) if avg_dur_sb > 0 else 2.0

        bpm, label = self._DENSITY_MAP[-1][1], self._DENSITY_MAP[-1][2]
        for max_npb, bpm_val, lbl in self._DENSITY_MAP:
            if notes_per_beat <= max_npb:
                bpm, label = bpm_val, lbl
                break

        beat_str = f"{beat.numerator}/{beat.denominator}"
        q_str    = f"Q:{beat_str}={bpm}"

        issues.append(Issue(
            line_index=k_idx,
            description=(
                f"未找到 Q: 速度标记；音符密度约每拍 {notes_per_beat:.1f} 个，"
                f"估算建议速度 {label}，建议添加 {q_str}"
            ),
            severity="info",
        ))

        if auto_fix:
            modified_lines.insert(k_idx, q_str)

        return issues, modified_lines


class CommentStripper(CheckerModule):
    """
    auto_fix=True 时清理输出：
      - 删除单 % 注释行（保留 %% 指令行）
      - 删除纯空白行（由其他模块合并音乐行后留下的占位行）
    检查模式下不产生任何 issue，直接透传。
    """

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        if not auto_fix:
            return [], list(lines)
        cleaned = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('%%'):
                cleaned.append(line)
            elif stripped.startswith('%'):
                continue
            elif not stripped:
                continue
            else:
                cleaned.append(line)
        return [], cleaned


class ABCProcessor:
    """
    核心处理引擎
    管理所有注册的模块并按顺序执行检查与修复管道
    """
    def __init__(self):
        self.modules: List[CheckerModule] = []

    def register_module(self, module: CheckerModule):
        self.modules.append(module)

    def run_pipeline(self, abc_content: str, auto_fix: bool = False) -> Tuple[List[Issue], str]:
        lines = abc_content.split("\n")
        all_issues = []

        for module in self.modules:
            issues, lines = module.process(lines, auto_fix)
            all_issues.extend(issues)

        final_content = "\n".join(lines)
        return all_issues, final_content


if __name__ == "__main__":
    sample_abc = """X:1
T:Rhythmic Logic Stress Test
C:Experimental
M:4/4
L:1/8
%%score {V:1 | V:2}
K:C
% --- 右手：包含连音符与超长时值 ---
V:1 nm="Piano" clef=treble
% 第一小节：正常 4/4 (8个1/8音符)
(3cde f2 (5g/a/g/f/e d2 |
% 第二小节：时值溢出 (包含了 10 个 1/8 音符的长度)
e4 (7g/a/b/c'/b/a/g f2 g2 a2 |
% 第三小节：时值缺失 (仅有 6 个 1/8 音符)
[CEG]4 d/e/f/ |
% 第四小节：复杂嵌套感
!trill!g6 (3abc' |]
% --- 左手：与右手时值完全不匹配 ---
V:2 clef=bass
% 第一小节：时值缺失 (仅 4 个 1/8 音符)
C,,2 G,,2 |
% 第二小节：时值正常，但与右手的溢出形成对比
F,,2 A,,2 C,2 E,2 |
% 第三小节：时值大幅溢出 (12 个 1/8 音符)
D,,8 E,,2 F,,2 G,,2 |
% 第四小节：极短小节
C,,1 |]
"""

    engine = ABCProcessor()

    # 规则和默认值通过字典动态配置
    header_rules = {
        "X": "1",
        "T": "Untitled",
        "M": "4/4",
        "L": "1/4",
    }

    engine.register_module(HeaderChecker(required_headers_with_defaults=header_rules))
    engine.register_module(TempoEstimator())
    engine.register_module(VoiceBarCountChecker())
    engine.register_module(MeasureDurationChecker())
    engine.register_module(ClefAutoSelector())
    engine.register_module(CommentStripper())

    print("执行纯检查模式：")
    issues, _ = engine.run_pipeline(sample_abc, auto_fix=False)
    for issue in issues:
        print(f"[{issue.severity}] 行号 {issue.line_index} -> {issue.description}")

    print("\n执行自动修复模式：")
    issues, fixed_abc = engine.run_pipeline(sample_abc, auto_fix=True)
    print("修复后的乐谱内容：")
    print(fixed_abc)
